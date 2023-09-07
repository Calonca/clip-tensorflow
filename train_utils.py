import tensorflow as tf

import utils
import CLIP_data_load
import numpy as np
from sklearn.metrics import accuracy_score
from wandb.keras import WandbCallback


checkpoint_fun = lambda filename: tf.keras.callbacks.ModelCheckpoint(filename, 
                    monitor="val_loss", mode="min", 
                    save_best_only=True, verbose=1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)

wandb_callback_fun = lambda gen: WandbCallback(
    monitor='loss',
    log_batch_frequency=10,
    save_model = False,
    validation_steps=len(gen),
)

## Loss
def tf_categorical_cross_entropy(y_true, logits):
    return tf.math.reduce_mean(
        tf.keras.metrics.categorical_crossentropy(
            y_true=y_true, y_pred=logits, from_logits=True
        )
    )


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
    text_logits = tf.matmul(text_embeds, image_embeds, transpose_b=True) / temperature
    image_logits = tf.transpose(text_logits)

    num_logits = image_logits.shape[0]

    labels = tf.range(num_logits)

    caption_loss = tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=labels, y_pred=text_logits, from_logits=True
        )
    )
    image_loss = tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=labels, y_pred=image_logits, from_logits=True
        )
    )

    loss = (caption_loss + image_loss) / 2.0
    if verbose:
        losses_dict = {}
        losses_dict["clip_loss"] = loss
        losses_dict["clip_caption_loss"] = caption_loss
        losses_dict["clip_image_loss"] = image_loss

        return losses_dict

    return loss


def custom_loss_with_temp(temp=1):
    return lambda text_embeds, image_embeds, verbose=False : custom_loss(text_embeds, image_embeds, temp, verbose)

"""Returns the our loss function with a given temperature"""
def custom_loss(text_embeds, image_embeds, temperature, verbose=False):
    """Uses images and text similarities"""
    image_embeds = image_embeds / tf.norm(tensor=image_embeds, axis=-1, keepdims=True)
    text_embeds = text_embeds / tf.norm(tensor=text_embeds, axis=-1, keepdims=True)

    logits = (
        tf.matmul(text_embeds, image_embeds, transpose_b=True) / temperature
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
        losses_dict["custom_loss"] = loss
        losses_dict["custom_caption_loss"] = img_loss
        losses_dict["custom_image_loss"] = txt_loss

        return losses_dict

    return loss


def get_hybrid_loss(
    temperature_base_loss=2,
    temperature_custom_loss=2,
    weight: float = 0.5,
    verbose_clip_loss=False,
    verbose_custom_loss=False,
):
    def hybrid_loss(text_embeds, image_embeds):
        # return c_loss + (1-weight)*cu_loss
        if verbose_clip_loss or verbose_custom_loss:
            losses_dict = {}

            if verbose_clip_loss:
                c_loss_dict = clip_loss(
                    text_embeds, image_embeds, temperature_base_loss, verbose_clip_loss
                )
                losses_dict.update(c_loss_dict)
                c_loss = c_loss_dict["clip_loss"]
            else:
                c_loss = clip_loss(text_embeds, image_embeds, temperature_base_loss)

            if verbose_custom_loss:
                cu_loss_dict = custom_loss(
                    text_embeds,
                    image_embeds,
                    temperature_base_loss,
                    verbose_custom_loss,
                )
                losses_dict.update(cu_loss_dict)
                cu_loss = cu_loss_dict["custom_loss"]
            else:
                cu_loss = custom_loss(
                    text_embeds, image_embeds, temperature_custom_loss
                )

            hybrid_loss = weight * c_loss + (1 - weight) * cu_loss

            losses_dict["hybrid_loss"] = hybrid_loss

            return losses_dict

        else:
            c_loss = clip_loss(text_embeds, image_embeds, temperature_base_loss)
            cu_loss = custom_loss(text_embeds, image_embeds, temperature_custom_loss)

            return c_loss * cu_loss

    return hybrid_loss


class MeanMetricClipLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = "mean_metric"
        super(MeanMetricClipLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(
            temperature_base_loss=0.8,
            temperature_custom_loss=0.5,
            weight=0.25,
            verbose_clip_loss=True,
        )
        self.clip_loss = self.add_weight(name="clip_loss_sum", initializer="zeros")
        self.clip_caption_loss = self.add_weight(
            name="clip_caption_loss_sum", initializer="zeros"
        )
        self.clip_image_loss = self.add_weight(
            name="clip_image_loss_sum", initializer="zeros"
        )
        self.loss = self.add_weight(name="hybrid_loss_sum", initializer="zeros")
        self.num_samples = self.add_weight(name="num_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.clip_loss.assign_add(values["clip_loss"] * batch_size)
        self.clip_caption_loss.assign_add(values["clip_caption_loss"] * batch_size)
        self.clip_image_loss.assign_add(values["clip_image_loss"] * batch_size)
        self.loss.assign_add(values["hybrid_loss"] * batch_size)
        self.num_samples.assign_add(batch_size)

    def result(self):
        mean_clip_loss = self.clip_loss / self.num_samples
        mean_clip_caption_loss = self.clip_caption_loss / self.num_samples
        mean_clip_image_loss = self.clip_image_loss / self.num_samples
        mean_loss = self.loss / self.num_samples

        return {
            "mean_clip_loss": mean_clip_loss,
            "mean_clip_caption_loss": mean_clip_caption_loss,
            "mean_clip_image_loss": mean_clip_image_loss,
            "mean_hybrid_loss": mean_loss,
        }

    def reset_states(self):
        self.clip_loss.assign(0.0)
        self.clip_caption_loss.assign(0.0)
        self.clip_image_loss.assign(0.0)
        self.loss.assign(0.0)
        self.num_samples.assign(0.0)


class LastValueMetricClipLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = "last_value_metric"
        super(LastValueMetricClipLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(
            temperature_base_loss=0.8,
            temperature_custom_loss=0.5,
            weight=0.25,
            verbose_clip_loss=True,
        )
        self.clip_loss = self.add_weight(name="clip_loss", initializer="zeros")
        self.clip_caption_loss = self.add_weight(
            name="clip_caption_loss", initializer="zeros"
        )
        self.clip_image_loss = self.add_weight(
            name="clip_image_loss", initializer="zeros"
        )
        self.loss = self.add_weight(name="hybrid_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        self.clip_loss.assign(values["clip_loss"])
        self.clip_caption_loss.assign(values["clip_caption_loss"])
        self.clip_image_loss.assign(values["clip_image_loss"])
        self.loss.assign(values["hybrid_loss"])

    def result(self):
        return {
            "clip_loss": self.clip_loss,
            "clip_caption_loss": self.clip_caption_loss,
            "clip_image_loss": self.clip_image_loss,
            "hybrid_loss": self.loss,
        }

    def reset_states(self):
        self.clip_loss.assign(0.0)
        self.clip_caption_loss.assign(0.0)
        self.clip_image_loss.assign(0.0)
        self.loss.assign(0.0)


class MeanMetricCustomLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = "mean_metric"
        super(MeanMetricCustomLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(
            temperature_base_loss=0.8,
            temperature_custom_loss=0.5,
            weight=0.25,
            verbose_custom_loss=True,
        )
        self.custom_loss = self.add_weight(name="custom_loss_sum", initializer="zeros")
        self.custom_caption_loss = self.add_weight(
            name="custom_caption_loss_sum", initializer="zeros"
        )
        self.custom_image_loss = self.add_weight(
            name="custom_image_loss_sum", initializer="zeros"
        )
        self.loss = self.add_weight(name="hybrid_loss", initializer="zeros")
        self.num_samples = self.add_weight(name="num_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.custom_loss.assign_add(values["custom_loss"] * batch_size)
        self.custom_caption_loss.assign_add(values["custom_caption_loss"] * batch_size)
        self.custom_image_loss.assign_add(values["custom_image_loss"] * batch_size)
        self.loss.assign_add(values["hybrid_loss"] * batch_size)
        self.num_samples.assign_add(batch_size)

    def result(self):
        mean_custom_loss = self.custom_loss / self.num_samples
        mean_custom_caption_loss = self.custom_caption_loss / self.num_samples
        mean_custom_image_loss = self.custom_image_loss / self.num_samples
        mean_loss = self.loss / self.num_samples

        return {
            "mean_custom_loss": mean_custom_loss,
            "mean_custom_caption_loss": mean_custom_caption_loss,
            "mean_custom_image_loss": mean_custom_image_loss,
            "mean_hybrid_loss": mean_loss,
        }

    def reset_states(self):
        self.custom_loss.assign(0.0)
        self.custom_caption_loss.assign(0.0)
        self.custom_image_loss.assign(0.0)
        self.loss.assign(0.0)
        self.num_samples.assign(0.0)


class LastValueMetricCustomLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = "last_value_metric"
        super(LastValueMetricCustomLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(
            temperature_base_loss=0.8,
            temperature_custom_loss=0.5,
            weight=0.25,
            verbose_custom_loss=True,
        )
        self.custom_loss = self.add_weight(name="custom_loss", initializer="zeros")
        self.custom_caption_loss = self.add_weight(
            name="custom_caption_loss", initializer="zeros"
        )
        self.custom_image_loss = self.add_weight(
            name="custom_image_loss", initializer="zeros"
        )
        self.loss = self.add_weight(name="hybrid_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        self.custom_loss.assign(values["custom_loss"])
        self.custom_caption_loss.assign(values["custom_caption_loss"])
        self.custom_image_loss.assign(values["custom_image_loss"])
        self.loss.assign(values["hybrid_loss"])

    def result(self):
        return {
            "custom_loss": self.custom_loss,
            "custom_caption_loss": self.custom_caption_loss,
            "custom_image_loss": self.custom_image_loss,
            "hybrid_loss": self.loss,
        }

    def reset_states(self):
        self.custom_loss.assign(0.0)
        self.custom_caption_loss.assign(0.0)
        self.custom_image_loss.assign(0.0)
        self.loss.assign(0.0)


class MeanMetricCombinedLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = "mean_metric_combined"
        super(MeanMetricCombinedLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(
            temperature_base_loss=0.8,
            temperature_custom_loss=0.5,
            weight=0.25,
            verbose_custom_loss=True,
            verbose_clip_Loss=True,
        )

        # Custom losses
        self.custom_loss = self.add_weight(name="custom_loss_sum", initializer="zeros")
        self.custom_caption_loss = self.add_weight(
            name="custom_caption_loss_sum", initializer="zeros"
        )
        self.custom_image_loss = self.add_weight(
            name="custom_image_loss_sum", initializer="zeros"
        )

        # Clip losses
        self.clip_loss = self.add_weight(name="clip_loss_sum", initializer="zeros")
        self.clip_caption_loss = self.add_weight(
            name="clip_caption_loss_sum", initializer="zeros"
        )
        self.clip_image_loss = self.add_weight(
            name="clip_image_loss_sum", initializer="zeros"
        )

        self.loss = self.add_weight(name="loss_sum", initializer="zeros")
        self.num_samples = self.add_weight(name="num_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)

        # Update custom losses
        self.custom_loss.assign_add(values["custom_loss"] * batch_size)
        self.custom_caption_loss.assign_add(values["custom_caption_loss"] * batch_size)
        self.custom_image_loss.assign_add(values["custom_image_loss"] * batch_size)

        # Update clip losses
        self.clip_loss.assign_add(values["clip_loss"] * batch_size)
        self.clip_caption_loss.assign_add(values["clip_caption_loss"] * batch_size)
        self.clip_image_loss.assign_add(values["clip_image_loss"] * batch_size)

        self.loss.assign_add(values["loss"] * batch_size)
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
            "mean_custom_loss": mean_custom_loss,
            "mean_custom_caption_loss": mean_custom_caption_loss,
            "mean_custom_image_loss": mean_custom_image_loss,
            "mean_clip_loss": mean_clip_loss,
            "mean_clip_caption_loss": mean_clip_caption_loss,
            "mean_clip_image_loss": mean_clip_image_loss,
            "mean_loss": mean_loss,
        }

    def reset_states(self):
        self.custom_loss.assign(0.0)
        self.custom_caption_loss.assign(0.0)
        self.custom_image_loss.assign(0.0)

        self.clip_loss.assign(0.0)
        self.clip_caption_loss.assign(0.0)
        self.clip_image_loss.assign(0.0)

        self.loss.assign(0.0)
        self.num_samples.assign(0.0)


class LastValueMetricCombinedLoss(tf.keras.metrics.Metric):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = "last_value_metric_combined"
        super(LastValueMetricCombinedLoss, self).__init__(name=name, **kwargs)
        self.fn = get_hybrid_loss(
            temperature_base_loss=0.8,
            temperature_custom_loss=0.5,
            weight=0.25,
            verbose_custom_loss=True,
            verbose_clip_Loss=True,
        )

        # Custom losses
        self.custom_loss = self.add_weight(name="custom_loss", initializer="zeros")
        self.custom_caption_loss = self.add_weight(
            name="custom_caption_loss", initializer="zeros"
        )
        self.custom_image_loss = self.add_weight(
            name="custom_image_loss", initializer="zeros"
        )

        # Clip losses
        self.clip_loss = self.add_weight(name="clip_loss", initializer="zeros")
        self.clip_caption_loss = self.add_weight(
            name="clip_caption_loss", initializer="zeros"
        )
        self.clip_image_loss = self.add_weight(
            name="clip_image_loss", initializer="zeros"
        )

        self.loss = self.add_weight(name="loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)

        # Update custom losses
        self.custom_loss.assign(values["custom_loss"])
        self.custom_caption_loss.assign(values["custom_caption_loss"])
        self.custom_image_loss.assign(values["custom_image_loss"])

        # Update clip losses
        self.clip_loss.assign(values["clip_loss"])
        self.clip_caption_loss.assign(values["clip_caption_loss"])
        self.clip_image_loss.assign(values["clip_image_loss"])

        self.loss.assign(values["hybrid_loss"])

    def result(self):
        return {
            "custom_loss": self.custom_loss,
            "custom_caption_loss": self.custom_caption_loss,
            "custom_image_loss": self.custom_image_loss,
            "clip_loss": self.clip_loss,
            "clip_caption_loss": self.clip_caption_loss,
            "clip_image_loss": self.clip_image_loss,
            "hybrid_loss": self.loss,
        }

    def reset_states(self):
        self.custom_loss.assign(0.0)
        self.custom_caption_loss.assign(0.0)
        self.custom_image_loss.assign(0.0)

        self.clip_loss.assign(0.0)
        self.clip_caption_loss.assign(0.0)
        self.clip_image_loss.assign(0.0)

        self.loss.assign(0.0)


"""Returns tokenized labels and one hot encoded labels"""


def build_zero_shot_metric(
    df, tokenizer, max_len_concepts, max_len_captions, preceding_caption, num_classes=10
):
    labels, weights = utils.get_most_common_labels(df, num_classes=num_classes)

    # one hot encode labels
    zs_labels = df.concepts.apply(
        lambda x: utils.one_hot_encode(x, labels, num_classes)
    )

    labels = [preceding_caption + label for label in labels]

    zs_concepts = CLIP_data_load.construct_encoding(
        labels, tokenizer, max_len_captions, return_tensors="tf"
    )
    zs_concepts_as_caps = CLIP_data_load.construct_encoding(
        labels, tokenizer, max_len_captions, return_tensors="tf"
    )
    return zs_concepts, zs_concepts_as_caps, zs_labels


class ZeroShotSingleLabelCallBack(tf.keras.callbacks.Callback):
    """similarity_treshold is an hyperparameter.
    Images that have similarity higher then the treshold will belong to the class"""

    def __init__(
        self,
        model,
        emb_fun,
        gen,
        zs_labels,
        logger,
        precomputed_e_img=None,
    ):
        self.model = model
        self.zs_labels = zs_labels
        self.emb_lambda = emb_fun
        self.gen = gen
        self.logger = logger
        self.precomputed_e_img = precomputed_e_img

    def on_epoch_end(self, epoch, metrics={}):

        label_e_txt = self.emb_lambda()

        # Skip embedding computation if embeddings previously computed
        if self.precomputed_e_img is not None:
            e_img = self.precomputed_e_img
        else:
            e_txt, e_img = self.model.predict(self.gen)

        one_hot_labels = np.vstack(np.array(self.zs_labels))
        one_hot_labels = one_hot_labels[
            0 : e_img.shape[0], :
        ]  # fix arbitrary batch size

        # get all the indexes of labels with 0 ones
        all_zero_indexes = np.where(np.sum(one_hot_labels, axis=1) == 0)
        # remove indexes with 0 ones from one_hot_labels and e_img
        one_hot_labels = np.delete(one_hot_labels, all_zero_indexes, axis=0)
        e_img = np.delete(e_img, all_zero_indexes, axis=0)
        # compute similarity between each image and label
        similarities = np.dot(e_img, label_e_txt.T)

        # sets predictions to 1 to each element that is >= 0.9 * max_similarity element
        # eg. [0.91, 0.5, 0.93, 0.92] -> [1, 0, 1, 1]
        similarities = (
            similarities >= similarities.max(axis=1, keepdims=1) * 0.9
        ).astype(float)

        # Multi class accuracy
        acc_score = accuracy_score(one_hot_labels, similarities)

        # return e_txt.numpy(),e_img.numpy()
        self.logger.log({"zero_shot_accuracy": acc_score})
        metrics["zero_shot_accuracy"] = acc_score #
