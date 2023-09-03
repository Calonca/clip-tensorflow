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
import tensorflow_addons as tfa

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

def c_loss(logits):
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


def get_hybrid_loss(
    temperature_base_loss=None, 
    temperature_custom_loss=0.5,
    weight:float=0.5,
    verbose_clip_loss=False,
    verbose_custom_loss=False):

    def hybrid_loss(text_embeds, image_embeds):

        
        #return c_loss + (1-weight)*cu_loss
        if verbose_clip_loss or verbose_custom_loss:

            losses_dict = {}

            if verbose_clip_loss:
                c_loss_dict = clip_loss(text_embeds, image_embeds, temperature_base_loss, verbose_clip_loss)
                losses_dict.update(c_loss_dict)
                c_loss = c_loss_dict['clip_loss']
            else:
                c_loss = clip_loss(text_embeds, image_embeds, temperature_base_loss)


            if verbose_custom_loss:
                cu_loss_dict = custom_loss(text_embeds, image_embeds, temperature_base_loss, verbose_custom_loss)
                losses_dict.update(cu_loss_dict)
                cu_loss = cu_loss_dict['custom_loss']
            else:
                cu_loss = custom_loss(text_embeds, image_embeds, temperature_custom_loss)

            hybrid_loss = c_loss * cu_loss

            losses_dict['hybrid_loss'] = hybrid_loss

            return losses_dict

        else:

            c_loss = clip_loss(text_embeds, image_embeds, temperature_base_loss)
            cu_loss = custom_loss(text_embeds, image_embeds, temperature_custom_loss)

            return c_loss * cu_loss

    return hybrid_loss


def clip_loss(text_embeds, image_embeds, temperature=None, verbose=False):

    if temperature is None:
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

    num_logits = image_logits.shape[0]

    labels=tf.range(num_logits)

    caption_loss = tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(y_true=labels, y_pred=text_logits, from_logits=True))
    image_loss = tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(y_true=labels, y_pred=image_logits, from_logits=True))


    loss = (caption_loss + image_loss) / 2.0
    if verbose:
        losses_dict = {}
        losses_dict['clip_loss'] = loss
        losses_dict['clip_caption_loss'] = caption_loss
        losses_dict['clip_image_loss'] = image_loss

        return losses_dict

    return loss

"""Returns the our loss function with a given temperature"""
def custom_loss(text_embeds, image_embeds, temperature, verbose=False):
    """Uses images and text similarities"""
    # image_embeds = image_embeds / tf.norm(tensor=image_embeds, axis=-1, keepdims=True)
    # text_embeds = text_embeds / tf.norm(tensor=text_embeds, axis=-1, keepdims=True)

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


    if verbose:
        losses_dict = {}
        losses_dict['custom_loss'] = loss
        losses_dict['custom_caption_loss'] = img_loss
        losses_dict['custom_image_loss'] = txt_loss

        return losses_dict

    return loss

def custom_loss_with_temp(temperature:float, verbose=False):

    if temperature is None:
        temperature=1.0

    """Returns the our loss function with a given temperature"""
    def custom_loss(text_embeds, image_embeds):
        """Uses images and text similarities"""
        # image_embeds = image_embeds / tf.norm(tensor=image_embeds, axis=-1, keepdims=True)
        # text_embeds = text_embeds / tf.norm(tensor=text_embeds, axis=-1, keepdims=True)
    
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
    
    
        if verbose:
            losses_dict = {}
            losses_dict['custom_loss'] = loss
            losses_dict['custom_caption_loss'] = img_loss
            losses_dict['custom_image_loss'] = txt_loss

            return losses_dict
        return loss
        
    return custom_loss

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
def get_projector(x, latent_dim, output_dim, dropout_rate):
    projected = tf.keras.layers.Dense(output_dim, activation="gelu")(x)
    x = tf.keras.layers.Dense(output_dim, activation="linear")(projected)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = x+projected
    x = tf.keras.layers.LayerNormalization()(x)
    return x

class ProjectorLayer(tf.keras.layers.Layer):
    def __init__(self, latent_dim, output_dim, dropout_rate, **kwargs):
        super(ProjectorLayer, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Define the layers
        self.dense_gelu = tf.keras.layers.Dense(self.output_dim, activation="gelu")
        self.dense_linear = tf.keras.layers.Dense(self.output_dim, activation="linear")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        projected = self.dense_gelu(inputs)
        x = self.dense_linear(projected)
        x = self.dropout(x)
        x = x + projected
        return self.layer_norm(x)

def t_f():
    return True

def f_f():
    return False

def get_clip_fusion_model(
    image_input_shape,
    concepts_input_shape,
    caption_input_shape,
    text_encoder,
    image_encoder,
    latent_dim_imgs,
    latent_dim_text,
    latent_dim_common,
    train_bert=False,
    projector_dropout_rate=0.1,
    loss=clip_loss,
    custom_metric = None,
    lr_image_encoder = 1e-4,
    lr_text_encoder = 1e-5,
    lr_head = 1e-3
)->tf.keras.Model:
    """Return a CLIP model

    Args:
        eval_loss: If not None the model will calculate the provided loss function as an additional metric
    """
    text_encoder.trainable = train_bert

    image_input = tf.keras.Input(shape=image_input_shape)
    caption_id_input = tf.keras.Input(shape=caption_input_shape, dtype=tf.int32)
    caption_mask_input = tf.keras.Input(shape=caption_input_shape)

    concepts_id_input = tf.keras.Input(shape=concepts_input_shape, dtype=tf.int32)
    concepts_mask_input = tf.keras.Input(shape=concepts_input_shape)

    caption_encoding = text_encoder(
            input_ids=caption_id_input, attention_mask=caption_mask_input
        ).last_hidden_state
    
    caption_encoding = caption_encoding[:, 0, :]

    non_zero_count = tf.math.reduce_sum(tf.cast(tf.math.not_equal(concepts_id_input, 0), tf.int32))
    
    check = tf.cond(tf.equal(non_zero_count, 0), t_f, f_f)

    concepts_encoding = text_encoder(
            input_ids=concepts_id_input, attention_mask=concepts_mask_input
        ).last_hidden_state

    concepts_encoding = concepts_encoding[:, 0, :]
    text_encoding = tf.concat(values=[caption_encoding, concepts_encoding], axis=1)

    image_encoding = image_encoder(image_input)
    # image_encoding = image_encoder(image_input).pooler_output
    # image_encoding = image_encoding[:,:,0,0]

    
    """print(caption_encoding)
    print(concepts_encoding)
    print(text_encoding)"""

    text_projector_name = "text_projector"
    image_projector_name = "image_projector"

    text_projector = ProjectorLayer(latent_dim_text, latent_dim_common, projector_dropout_rate, name=text_projector_name)(text_encoding)
    image_projector = ProjectorLayer(latent_dim_imgs, latent_dim_common, projector_dropout_rate, name=image_projector_name)(image_encoding)

    """print(text_projector)
    print(image_projector)"""

    image_projector = tf.squeeze(image_projector)

    if custom_metric is None:
        model = CLIP_fusion(
        inputs=[image_input, caption_id_input, caption_mask_input, concepts_id_input, concepts_mask_input],
        outputs=[text_projector, image_projector],
        
        )
    else:
        model = CLIP_fusion_with_custom_metric(
            inputs=[image_input, caption_id_input, caption_mask_input, concepts_id_input, concepts_mask_input],
            outputs=[text_projector, image_projector],
            custom_metric_tracker=custom_metric
        )


    image_encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_image_encoder)
    text_encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_text_encoder)
    text_head_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_head)
    image_head_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_head)

    optimizers = [image_encoder_optimizer,text_encoder_optimizer, text_head_optimizer, image_head_optimizer]

    optimizers_and_layers = [
        (optimizers[0], model.get_layer(image_encoder.name)), 
        (optimizers[1], model.get_layer(text_encoder.name)),
        (optimizers[2], model.get_layer(text_projector_name)),
        (optimizers[3], model.get_layer(image_projector_name))
        ]
    
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, run_eagerly=True)

    return model

class MeanMetricClipLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = 'mean_metric'
        super(MeanMetricClipLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(temperature_base_loss=0.8, temperature_custom_loss=0.5, weight=0.25, verbose_clip_loss=True)
        self.clip_loss = self.add_weight(name="clip_loss_sum", initializer="zeros")
        self.clip_caption_loss = self.add_weight(name="clip_caption_loss_sum", initializer="zeros")
        self.clip_image_loss = self.add_weight(name="clip_image_loss_sum", initializer="zeros")
        self.loss = self.add_weight(name="hybrid_loss_sum", initializer="zeros")
        self.num_samples = self.add_weight(name="num_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.clip_loss.assign_add(values['clip_loss'] * batch_size)
        self.clip_caption_loss.assign_add(values['clip_caption_loss'] * batch_size)
        self.clip_image_loss.assign_add(values['clip_image_loss'] * batch_size)
        self.loss.assign_add(values['hybrid_loss'] * batch_size)
        self.num_samples.assign_add(batch_size)

    def result(self):
        mean_clip_loss = self.clip_loss / self.num_samples
        mean_clip_caption_loss = self.clip_caption_loss / self.num_samples
        mean_clip_image_loss = self.clip_image_loss / self.num_samples
        mean_loss = self.loss / self.num_samples

        return {
            'mean_clip_loss': mean_clip_loss,
            'mean_clip_caption_loss': mean_clip_caption_loss,
            'mean_clip_image_loss': mean_clip_image_loss,
            'mean_hybrid_loss': mean_loss
        }

    def reset_states(self):
        self.clip_loss.assign(0.)
        self.clip_caption_loss.assign(0.)
        self.clip_image_loss.assign(0.)
        self.loss.assign(0.)
        self.num_samples.assign(0.)

class LastValueMetricClipLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = 'last_value_metric'
        super(LastValueMetricClipLoss, self).__init__(name=name, **kwargs)
        self.fn =  get_hybrid_loss(temperature_base_loss=0.8, temperature_custom_loss=0.5, weight=0.25, verbose_clip_loss=True)
        self.clip_loss = self.add_weight(name="clip_loss", initializer="zeros")
        self.clip_caption_loss = self.add_weight(name="clip_caption_loss", initializer="zeros")
        self.clip_image_loss = self.add_weight(name="clip_image_loss", initializer="zeros")
        self.loss = self.add_weight(name="hybrid_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        self.clip_loss.assign(values['clip_loss'])
        self.clip_caption_loss.assign(values['clip_caption_loss'])
        self.clip_image_loss.assign(values['clip_image_loss'])
        self.loss.assign(values['hybrid_loss'])

    def result(self):
        return {
            'clip_loss': self.clip_loss,
            'clip_caption_loss': self.clip_caption_loss,
            'clip_image_loss': self.clip_image_loss,
            'hybrid_loss': self.loss
        }

    def reset_states(self):
        self.clip_loss.assign(0.)
        self.clip_caption_loss.assign(0.)
        self.clip_image_loss.assign(0.)
        self.loss.assign(0.)

class MeanMetricCustomLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = 'mean_metric'
        super(MeanMetricCustomLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(temperature_base_loss=0.8, temperature_custom_loss=0.5, weight=0.25, verbose_custom_loss=True)
        self.custom_loss = self.add_weight(name="custom_loss_sum", initializer="zeros")
        self.custom_caption_loss = self.add_weight(name="custom_caption_loss_sum", initializer="zeros")
        self.custom_image_loss = self.add_weight(name="custom_image_loss_sum", initializer="zeros")
        self.loss = self.add_weight(name="hybrid_loss", initializer="zeros")
        self.num_samples = self.add_weight(name="num_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.custom_loss.assign_add(values['custom_loss'] * batch_size)
        self.custom_caption_loss.assign_add(values['custom_caption_loss'] * batch_size)
        self.custom_image_loss.assign_add(values['custom_image_loss'] * batch_size)
        self.loss.assign_add(values['hybrid_loss'] * batch_size)
        self.num_samples.assign_add(batch_size)

    def result(self):
        mean_custom_loss = self.custom_loss / self.num_samples
        mean_custom_caption_loss = self.custom_caption_loss / self.num_samples
        mean_custom_image_loss = self.custom_image_loss / self.num_samples
        mean_loss = self.loss / self.num_samples

        return {
            'mean_custom_loss': mean_custom_loss,
            'mean_custom_caption_loss': mean_custom_caption_loss,
            'mean_custom_image_loss': mean_custom_image_loss,
            'mean_hybrid_loss': mean_loss
        }

    def reset_states(self):
        self.custom_loss.assign(0.)
        self.custom_caption_loss.assign(0.)
        self.custom_image_loss.assign(0.)
        self.loss.assign(0.)
        self.num_samples.assign(0.)

class LastValueMetricCustomLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = 'last_value_metric'
        super(LastValueMetricCustomLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(temperature_base_loss=0.8, temperature_custom_loss=0.5, weight=0.25, verbose_custom_loss=True)
        self.custom_loss = self.add_weight(name="custom_loss", initializer="zeros")
        self.custom_caption_loss = self.add_weight(name="custom_caption_loss", initializer="zeros")
        self.custom_image_loss = self.add_weight(name="custom_image_loss", initializer="zeros")
        self.loss = self.add_weight(name="hybrid_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        self.custom_loss.assign(values['custom_loss'])
        self.custom_caption_loss.assign(values['custom_caption_loss'])
        self.custom_image_loss.assign(values['custom_image_loss'])
        self.loss.assign(values['hybrid_loss'])

    def result(self):
        return {
            'custom_loss': self.custom_loss,
            'custom_caption_loss': self.custom_caption_loss,
            'custom_image_loss': self.custom_image_loss,
            'hybrid_loss': self.loss
        }

    def reset_states(self):
        self.custom_loss.assign(0.)
        self.custom_caption_loss.assign(0.)
        self.custom_image_loss.assign(0.)
        self.loss.assign(0.)

class MeanMetricCombinedLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = 'mean_metric_combined'
        super(MeanMetricCombinedLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(temperature_base_loss=0.8, temperature_custom_loss=0.5, weight=0.25, verbose_custom_loss=True, verbose_clip_Loss=True)
        
        # Custom losses
        self.custom_loss = self.add_weight(name="custom_loss_sum", initializer="zeros")
        self.custom_caption_loss = self.add_weight(name="custom_caption_loss_sum", initializer="zeros")
        self.custom_image_loss = self.add_weight(name="custom_image_loss_sum", initializer="zeros")
        
        # Clip losses
        self.clip_loss = self.add_weight(name="clip_loss_sum", initializer="zeros")
        self.clip_caption_loss = self.add_weight(name="clip_caption_loss_sum", initializer="zeros")
        self.clip_image_loss = self.add_weight(name="clip_image_loss_sum", initializer="zeros")
        
        self.loss = self.add_weight(name="loss_sum", initializer="zeros")
        self.num_samples = self.add_weight(name="num_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        
        # Update custom losses
        self.custom_loss.assign_add(values['custom_loss'] * batch_size)
        self.custom_caption_loss.assign_add(values['custom_caption_loss'] * batch_size)
        self.custom_image_loss.assign_add(values['custom_image_loss'] * batch_size)
        
        # Update clip losses
        self.clip_loss.assign_add(values['clip_loss'] * batch_size)
        self.clip_caption_loss.assign_add(values['clip_caption_loss'] * batch_size)
        self.clip_image_loss.assign_add(values['clip_image_loss'] * batch_size)
        
        self.loss.assign_add(values['loss'] * batch_size)
        self.num_samples.assign_add(batch_size)

    def result(self):
        # Mean custom losses
        mean_custom_loss = self.custom_loss / self.num_samples
        mean_custom_caption_loss = self.custom_caption_loss / self.num_samples
        mean_custom_image_loss = self.custom_image_loss / self.num_samples
        
        # Mean clip losses
        mean_clip_loss = self.clip_loss / self.num_samples
        mean_clip_caption_loss = self.clip_caption_loss / self.num_samples
        mean_clip_image_loss = self.clip_image_loss / self.num_samples
        
        mean_loss = self.loss / self.num_samples

        return {
            'mean_custom_loss': mean_custom_loss,
            'mean_custom_caption_loss': mean_custom_caption_loss,
            'mean_custom_image_loss': mean_custom_image_loss,
            'mean_clip_loss': mean_clip_loss,
            'mean_clip_caption_loss': mean_clip_caption_loss,
            'mean_clip_image_loss': mean_clip_image_loss,
            'mean_loss': mean_loss
        }

    def reset_states(self):
        self.custom_loss.assign(0.)
        self.custom_caption_loss.assign(0.)
        self.custom_image_loss.assign(0.)
        
        self.clip_loss.assign(0.)
        self.clip_caption_loss.assign(0.)
        self.clip_image_loss.assign(0.)
        
        self.loss.assign(0.)
        self.num_samples.assign(0.)

class LastValueMetricCombinedLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = 'last_value_metric_combined'
        super(LastValueMetricCombinedLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(temperature_base_loss=0.8, temperature_custom_loss=0.5, weight=0.25, verbose_custom_loss=True, verbose_clip_Loss=True)
        
        # Custom losses
        self.custom_loss = self.add_weight(name="custom_loss", initializer="zeros")
        self.custom_caption_loss = self.add_weight(name="custom_caption_loss", initializer="zeros")
        self.custom_image_loss = self.add_weight(name="custom_image_loss", initializer="zeros")
        
        # Clip losses
        self.clip_loss = self.add_weight(name="clip_loss", initializer="zeros")
        self.clip_caption_loss = self.add_weight(name="clip_caption_loss", initializer="zeros")
        self.clip_image_loss = self.add_weight(name="clip_image_loss", initializer="zeros")
        
        self.loss = self.add_weight(name="loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        
        # Update custom losses
        self.custom_loss.assign(values['custom_loss'])
        self.custom_caption_loss.assign(values['custom_caption_loss'])
        self.custom_image_loss.assign(values['custom_image_loss'])
        
        # Update clip losses
        self.clip_loss.assign(values['clip_loss'])
        self.clip_caption_loss.assign(values['clip_caption_loss'])
        self.clip_image_loss.assign(values['clip_image_loss'])
        
        self.loss.assign(values['hybrid_loss'])

    def result(self):
        return {
            'custom_loss': self.custom_loss,
            'custom_caption_loss': self.custom_caption_loss,
            'custom_image_loss': self.custom_image_loss,
            'clip_loss': self.clip_loss,
            'clip_caption_loss': self.clip_caption_loss,
            'clip_image_loss': self.clip_image_loss,
            'hybrid_loss': self.loss
        }

    def reset_states(self):
        self.custom_loss.assign(0.)
        self.custom_caption_loss.assign(0.)
        self.custom_image_loss.assign(0.)
        
        self.clip_loss.assign(0.)
        self.clip_caption_loss.assign(0.)
        self.clip_image_loss.assign(0.)
        
        self.loss.assign(0.)

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
    custom_metric = None
)->tf.keras.Model:
    """Return a CLIP model

    Args:
        eval_loss: If not None the model will calculate the provided loss function as an additional metric
    """
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

    if custom_metric is None:
        model = CLIP_base(
            inputs=[image_input, text_id_input, text_mask_input],
            outputs=[text_projector, image_projector]
        )
    else:
        model = CLIP_with_custom_metric(
            inputs=[image_input, text_id_input, text_mask_input],
            outputs=[text_projector, image_projector],
            custom_metric=custom_metric
        )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, run_eagerly=True)

    return model

class CLIP_fusion(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        imgs, caption_ids, caption_masks, concepts_ids, concepts_masks = data[0], data[1], data[2], data[3], data[4]

        with tf.GradientTape() as tape:
            text_projector, image_projector = self(
                [imgs, caption_ids, caption_masks, concepts_ids, concepts_masks], training=True
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
        imgs, caption_ids, caption_masks, concepts_ids, concepts_masks = data[0], data[1], data[2], data[3], data[4]
        # Compute predictions
        text_projector, image_projector = self(
            [imgs, caption_ids, caption_masks, concepts_ids, concepts_masks], training=False
        )  # Forward pass
        # Updates the metrics tracking the loss
        loss = self.compiled_loss(
            text_projector, image_projector, regularization_losses=self.losses
        )
        # Update the metrics.

    
        return {"loss": loss}

    def predict_step(self, data):

        if len(data)== 3:
            imgs, caption_ids, caption_masks = data[0], data[1], data[2]
            text_embeds, image_embeds = self(
                [imgs, caption_ids, caption_masks, None, None], training=False
            )  # Forward pass
        else: 
            imgs, caption_ids, caption_masks, concepts_ids, concepts_masks = data[0], data[1], data[2], data[3], data[4]

            text_embeds, image_embeds = self(
                [imgs, caption_ids, caption_masks, concepts_ids, concepts_masks], training=False
            )  # Forward pass
        
        image_embeds = image_embeds / tf.norm(
            tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True
        )
        text_embeds = text_embeds / tf.norm(
            tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True
        )

        return text_embeds, image_embeds

class CLIP_fusion_with_custom_metric(tf.keras.Model):
    def __init__(self,custom_metric_tracker, *args, **kwargs):
        super(CLIP_fusion_with_custom_metric, self).__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.custom_metric_tracker = custom_metric_tracker
        self.reduced_input = False

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        imgs, caption_ids, caption_masks, concepts_ids, concepts_masks = data[0], data[1], data[2], data[3], data[4]

        with tf.GradientTape() as tape:
            text_projector, image_projector = self(
                [imgs, caption_ids, caption_masks, concepts_ids, concepts_masks], training=True
            )  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)

            loss = self.compute_loss(text_projector, image_projector)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(text_projector, image_projector)

        # Return a dict mapping metric names to current value
        results = {}
        for m in self.metrics:
            res = m.result()

            if isinstance(res,dict):
                results.update(res)
            else:
                results[m.name] = res


        return results

    def test_step(self, data):
        # Unpack the data
        imgs, caption_ids, caption_masks, concepts_ids, concepts_masks = data[0], data[1], data[2], data[3], data[4]
        # Compute predictions
        text_projector, image_projector = self(
            [imgs, caption_ids, caption_masks, concepts_ids, concepts_masks], training=False
        )  # Forward pass
        # Updates the metrics tracking the loss
        loss = self.compute_loss(text_projector, image_projector)
         # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(text_projector, image_projector)
        # Return a dict mapping metric names to current value
        results = {}
        for m in self.metrics:
            res = m.result()

            if isinstance(res,dict):
                results.update(res)
            else:
                results[m.name] = res
        
        return results

    def predict_step(self, data):

        if len(data) == 3:
            self.reduce_input = True
            imgs, caption_ids, caption_masks = data[0], data[1], data[2]

            text_embeds, image_embeds = self(
                [imgs, caption_ids, caption_masks, tf.zeros_like(self.input_shape[-1]),tf.zeros_like(self.input_shape[-1])], training=False
            )  # Forward pass
        else: 
            imgs, caption_ids, caption_masks, concepts_ids, concepts_masks = data[0], data[1], data[2], data[3], data[4]

            text_embeds, image_embeds = self(
                [imgs, caption_ids, caption_masks, concepts_ids, concepts_masks], training=False
            )  # Forward pass
        
        image_embeds = image_embeds / tf.norm(
            tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True
        )
        text_embeds = text_embeds / tf.norm(
            tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True
        )

        self.reduce_input = False

        return text_embeds, image_embeds

    def compute_loss(self, text_projector, image_projector):
        loss = self.compiled_loss(
            text_projector, image_projector, regularization_losses=self.losses
        )
        self.loss_tracker.update_state(loss)
        return loss   

    def reset_metrics(self):
        self.loss_tracker.reset_states()
        self.custom_metric_tracker.reset_states()

    @property
    def metrics(self):
        return [self.loss_tracker, self.custom_metric_tracker]

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

class CLIP_with_custom_metric(tf.keras.Model):
    """Same architecture as CLIP_Base but computes the given custom metric in addition to the loss.
    https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L1157-L1211"""
    def __init__(self,custom_metric, *args, **kwargs):
        super(CLIP_with_custom_metric, self).__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.custom_metric_tracker = tf.keras.metrics.MeanMetricWrapper(name=custom_metric.__name__,fn=custom_metric)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        imgs, ids, masks = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            text_projector_o, image_projector_o = self(
                [imgs, ids, masks], training=True
            )  # Forward pass
            loss = self.compute_loss(text_projector_o, image_projector_o)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Updates the metrics tracking the loss
        loss = self.compute_loss(text_projector_o, image_projector_o)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(text_projector_o, image_projector_o)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        imgs, ids, masks = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Compute predictions
        text_projector_o, image_projector_o = self(
            [imgs, ids, masks], training=False
        )  # Forward pass

        # Updates the metrics tracking the loss
        loss = self.compute_loss(text_projector_o, image_projector_o)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(text_projector_o, image_projector_o)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

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
    
    def compute_loss(self, text_projector_o, image_projector_o):
        loss = self.compiled_loss(
            text_projector_o, image_projector_o, regularization_losses=self.losses
        )
        self.loss_tracker.update_state(loss)
        return loss   

    def reset_metrics(self):
        self.loss_tracker.reset_states()
        self.custom_metric_tracker.reset_states()

    @property
    def metrics(self):
        return [self.loss_tracker, self.custom_metric_tracker]