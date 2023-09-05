# Hyperparameters
train_percentage = 0.8
test_percentage = 0.2
validation_percentage = 0.15
latent_dim_imgs = 1024
latent_dim_text = 768
latent_dim_common = 512
lr_image_encoder = 1e-3
lr_text_encoder = 1e-5
lr_projector = 1e-4
projector_dropout_rate = 0.1
batch_size = 80
SEED = 116
img_shape = (128, 128, 3)
captions_input_shape = (128,)
concepts_input_shape = (80,)
zs_num_classes = 4  # number of classes to classify for zero shot learning
