import os
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import matplotlib as mpl
from transformers import BertConfig, BertTokenizer, BertModel
import matplotlib.pyplot as plt
from keras.callbacks import *
import random
from keras import backend as K
from PIL import Image
import shutil
import scipy
from tqdm import tqdm
import albumentations as A

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tfk = tf.keras
tfkl = tf.keras.layers

## Loss
def contrastive_loss(logits) :
      return tf.math.reduce_mean(
          tf.keras.metrics.sparse_categorical_crossentropy(
              y_true=tf.range(logits.shape[0]), y_pred=logits, from_logits=True
          )
      )

def clip_loss(text_embeds, image_embeds, logit_scale=None):

    logit_scale = 1.0
    
    # normalized feature
    image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True)
    text_embeds = text_embeds / tf.norm(tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True)

    logit_scale = tf.math.exp(logit_scale)
    text_logits = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
    image_logits = tf.transpose(text_logits)
    
    caption_loss = contrastive_loss(text_logits)
    image_loss = contrastive_loss(image_logits)

    return (caption_loss + image_loss) / 2.0

#Utils

def get_projector(x, latent_dim, output_dim):
    x = tf.keras.layers.Dense(latent_dim, activation='gelu')(x)
    x = tf.keras.layers.Dense(output_dim, activation='linear')(x)
    return x


def get_clip_model(
    image_input_shape,
    text_input_shape,
    text_encoder,
    image_encoder,
    latent_dim_imgs,
    latent_dim_text,
    latent_dim_common,
    train_bert = False
  ):

  text_encoder.trainable = train_bert
  
  image_input = tf.keras.Input(shape=image_input_shape)
  text_id_input = tf.keras.Input(shape=text_input_shape, dtype=tf.int32)
  text_mask_input = tf.keras.Input(shape=text_input_shape)

  text_encoding = text_encoder(input_ids = text_id_input, attention_mask = text_mask_input).last_hidden_state

  text_encoding = text_encoding[:, 0, :]
  image_encoding = image_encoder(image_input)

  text_projector = get_projector(text_encoding, latent_dim_text, latent_dim_common)
  image_projector = get_projector(image_encoding, latent_dim_imgs, latent_dim_common)

  image_projector = tf.squeeze(image_projector)


  model = CLIP_base(inputs=[image_input, text_id_input, text_mask_input], outputs = [text_projector, image_projector])
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
  model.compile(loss=clip_loss, optimizer=optimizer, run_eagerly=True)

  return model

class CLIP_base(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        imgs, ids, masks = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            text_projector, image_projector = self([imgs, ids, masks], training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)

            loss = self.compiled_loss(text_projector, image_projector, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # Return a dict mapping metric names to current value
        return {'loss': loss}
    
    def test_step(self, data):
                
        # Unpack the data
        imgs, ids, masks = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Compute predictions
        text_projector, image_projector = self([imgs, ids, masks], training=False)  # Forward pass
        # Updates the metrics tracking the loss
        loss = self.compiled_loss(text_projector, image_projector, regularization_losses=self.losses)
        # Update the metrics.
        return {'loss': loss}
    
    def predict_step(self, data):
        imgs, ids, masks = tf.keras.utils.unpack_x_y_sample_weight(data)

        # Compute predictions
        text_embeds, image_embeds = self([imgs, ids, masks], training=False)  # Forward pass
        
        image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True)

        return text_embeds, image_embeds
    