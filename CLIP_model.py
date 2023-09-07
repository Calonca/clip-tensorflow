import tensorflow as tf
import tensorflow_addons as tfa
from train_utils import *
from keras.optimizers import Adam

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tfk = tf.keras
tfkl = tf.keras.layers

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

    def call(self, inputs):
        projected = self.dense_gelu(inputs)
        x = self.dense_linear(projected)
        x = self.dropout(x)
        x = x + projected
        return x

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
    lr_projector = 1e-3
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

    concepts_encoding = text_encoder(
            input_ids=concepts_id_input, attention_mask=concepts_mask_input
        ).last_hidden_state

    concepts_encoding = concepts_encoding[:, 0, :]
    text_encoding = tf.concat(values=[caption_encoding, concepts_encoding], axis=1)

    image_encoding = image_encoder(image_input)

    text_projector_name = "text_projector"
    image_projector_name = "image_projector"

    text_projector = ProjectorLayer(latent_dim_text, latent_dim_common, projector_dropout_rate, name=text_projector_name)(text_encoding)
    image_projector = ProjectorLayer(latent_dim_imgs, latent_dim_common, projector_dropout_rate, name=image_projector_name)(image_encoding)

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

    optimizers_and_layers = [
        (Adam(learning_rate=lr_image_encoder), model.get_layer(image_encoder.name)), 
        (Adam(learning_rate=lr_text_encoder), model.get_layer(text_encoder.name)),
        (Adam(learning_rate=lr_projector), model.get_layer(text_projector_name)),
        (Adam(learning_rate=lr_projector), model.get_layer(image_projector_name))
        ]

    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    model.compile(loss=loss, optimizer=optimizer, run_eagerly=True)

    return model

def get_clip_model(
    image_input_shape,
    text_input_shape,
    text_encoder,
    image_encoder,
    latent_dim_imgs,
    latent_dim_text,
    latent_dim_common,
    train_bert=False,
    projector_dropout_rate=0.1,
    lr_image_encoder = 1e-4,
    lr_text_encoder = 1e-5,
    lr_projector = 1e-3,
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

    text_projector_name = "text_projector"
    image_projector_name = "image_projector"

    text_projector = ProjectorLayer(latent_dim_text, latent_dim_common, projector_dropout_rate, name=text_projector_name)(text_encoding)
    image_projector = ProjectorLayer(latent_dim_imgs, latent_dim_common, projector_dropout_rate, name=image_projector_name)(image_encoding)

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
            custom_metric_tracker=custom_metric
        )

    optimizers_and_layers = [
        (Adam(learning_rate=lr_image_encoder), model.get_layer(image_encoder.name)), 
        (Adam(learning_rate=lr_text_encoder), model.get_layer(text_encoder.name)),
        (Adam(learning_rate=lr_projector), model.get_layer(text_projector_name)),
        (Adam(learning_rate=lr_projector), model.get_layer(image_projector_name))
        ]

    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    
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

        imgs, ids, masks = data[0], data[1], data[2]

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
        imgs, ids, masks = data[0], data[1], data[2]
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
        imgs, ids, masks = data[0], data[1], data[2]

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
    def __init__(self,custom_metric_tracker, *args, **kwargs):
        super(CLIP_with_custom_metric, self).__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.custom_metric_tracker = custom_metric_tracker

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        imgs, ids, masks = data[0], data[1], data[2]

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
        imgs, ids, masks = data[0], data[1], data[2]
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
        imgs, ids, masks = data[0], data[1], data[2]

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
