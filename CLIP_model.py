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
def contrastive_loss(logits):
    return tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(  # we use sparse because we have integer labels
            y_true=tf.range(logits.shape[0]),  # labels from 0 to batch_size
            y_pred=logits,
            from_logits=True,
        )
    )


## Loss
def tf_categorical_cross_entropy(y_true, logits):
    return tf.math.reduce_mean(
        tf.keras.metrics.categorical_crossentropy(
            y_true=y_true, y_pred=logits, from_logits=True
        )
    )


def clip_loss(text_embeds, image_embeds, temperature=None):
    temperature = 1.0

    # normalized feature
    image_embeds = image_embeds / tf.norm(
        tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True
    )
    text_embeds = text_embeds / tf.norm(
        tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True
    )

    temperature = tf.math.exp(temperature)
    text_logits = tf.matmul(text_embeds, image_embeds, transpose_b=True) * temperature
    image_logits = tf.transpose(text_logits)

    caption_loss = contrastive_loss(text_logits)
    image_loss = contrastive_loss(image_logits)

    return (caption_loss + image_loss) / 2.0


def our_loss(text_embeds, image_embeds, temperature=1):
    """Uses images and text similarities"""
    image_embeds = image_embeds / tf.norm(tensor=image_embeds, axis=-1, keepdims=True)
    text_embeds = text_embeds / tf.norm(tensor=text_embeds, axis=-1, keepdims=True)

    logits = (
        tf.matmul(text_embeds, image_embeds, transpose_b=True) * temperature
    )  # rows are text and columns are images

    img_sim = tf.matmul(image_embeds, image_embeds, transpose_b=True)
    txt_sim = tf.matmul(text_embeds, text_embeds, transpose_b=True)

    text_true = tf.nn.softmax(
        ((img_sim + txt_sim) / 2.0) / temperature,
        axis=1,
    )
    img_true = tf.nn.softmax(
        ((img_sim + txt_sim) / 2.0) / temperature,
        axis=0,
    )

    img_loss = tf_categorical_cross_entropy(img_true, tf.transpose(logits))
    txt_loss = tf_categorical_cross_entropy(text_true, logits)
    loss = (img_loss + txt_loss) / 2.0
    return tf.reduce_mean(loss)


def loose_loss(text_embeds, image_embeds, temperature=1):
    """
    Inspired by https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py#L37
    Uses less strict loss function that will penalize less if there are similar captions or images in the batch.
    Both text and image embeds are (batch size,embedding size) dimensional vectors.
    """
    # normalize embeddings vectors
    # image_embeds = image_embeds / tf.norm(tensor=image_embeds, axis=-1, keepdims=True)
    # text_embeds = text_embeds / tf.norm(tensor=text_embeds, axis=-1, keepdims=True)
    logits = tf.matmul(text_embeds, image_embeds, transpose_b=True) / temperature
    img_sim = tf.matmul(image_embeds, image_embeds, transpose_b=True)
    txt_sim = tf.matmul(text_embeds, text_embeds, transpose_b=True)

    y_true = tf.nn.softmax(
        ((img_sim + txt_sim) / 2.0) * temperature,
        axis=-1,
    )

    texts_loss = cross_entropy(y_true, logits, reduction="none")
    images_loss = cross_entropy(
        tf.transpose(y_true), tf.transpose(logits), reduction="none"
    )
    loss = (images_loss + texts_loss) / 2.0
    return tf.reduce_mean(loss)


def cross_entropy(targets, preds, reduction="none"):
    loss = -targets * tf.reduce_sum(tf.nn.log_softmax(preds, axis=-1), axis=1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return tf.reduce_mean(loss)


# Utils
def get_projector(x, latent_dim, output_dim):
    x = tf.keras.layers.Dense(latent_dim, activation="gelu")(x)
    x = tf.keras.layers.Dense(output_dim, activation="linear")(x)
    return x


def get_clip_model(
    image_input_shape,
    text_input_shape,
    text_encoder,
    image_encoder,
    latent_dim_imgs,
    latent_dim_text,
    latent_dim_common,
    train_bert=False,
    learning_rate=1e-5,
    loss=clip_loss,
):
    text_encoder.trainable = train_bert

    image_input = tf.keras.Input(shape=image_input_shape)
    text_id_input = tf.keras.Input(shape=text_input_shape, dtype=tf.int32)
    text_mask_input = tf.keras.Input(shape=text_input_shape)

    text_encoding = text_encoder(
        input_ids=text_id_input, attention_mask=text_mask_input
    ).last_hidden_state

    text_encoding = text_encoding[:, 0, :]
    image_encoding = image_encoder(image_input)
    # image_encoding = image_encoder(image_input).pooler_output
    # image_encoding = image_encoding[:,:,0,0]

    text_projector = get_projector(text_encoding, latent_dim_text, latent_dim_common)
    image_projector = get_projector(image_encoding, latent_dim_imgs, latent_dim_common)

    image_projector = tf.squeeze(image_projector)

    model = CLIP_base(
        inputs=[image_input, text_id_input, text_mask_input],
        outputs=[text_projector, image_projector],
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, run_eagerly=True)

    return model


class CLIP_base(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        imgs, ids, masks = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            text_projector, image_projector = self(
                [imgs, ids, masks], training=True
            )  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)

            loss = self.compiled_loss(
                text_projector, image_projector, regularization_losses=self.losses
            )

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)
            # Return a dict mapping metric names to current value
            return {"loss": loss}

    def test_step(self, data):
        # Unpack the data
        imgs, ids, masks = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Compute predictions
        text_projector, image_projector = self(
            [imgs, ids, masks], training=False
        )  # Forward pass
        # Updates the metrics tracking the loss
        loss = self.compiled_loss(
            text_projector, image_projector, regularization_losses=self.losses
        )
        # Update the metrics.
        return {"loss": loss}

    def predict_step(self, data):
        imgs, ids, masks = tf.keras.utils.unpack_x_y_sample_weight(data)

        # Compute predictions
        text_embeds, image_embeds = self(
            [imgs, ids, masks], training=False
        )  # Forward pass

        image_embeds = image_embeds / tf.norm(
            tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True
        )
        text_embeds = text_embeds / tf.norm(
            tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True
        )

        return text_embeds, image_embeds
