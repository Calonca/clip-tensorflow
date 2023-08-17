import os

import cv2
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm


class ClipBaseGenerator(tf.keras.utils.Sequence):
  """
    CustomGenerator inheriting from tf.keras.utils.Sequence.

    We have to implement 3 main methods:
      - __init__: save dataset params like directory, filenames, etc.
      - __len__: return the total number of samples in the dataset (number of batches)
      - __getitem__: return a single batch of paired images masks
  """

  def __init__(self, 
               data,
               preprocessing_function=None, # Preprocessing function (e.g., the one used for transfer learning)
               batch_size=16, # Batch size
               out_shape = (100,100),
               shuffle=False,
               categorical = True,
               augment = False,
               seed=116,
               preprocess_input = False,
               channels_first=False):


    self.data = data
    self.indices = np.arange(len(self.data))
    # Save dataset parameters as class attributes
    self.preprocessing_function = preprocessing_function
    self.out_shape = out_shape
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.augment = augment
    self.seed = seed
    self.channels_first = channels_first

  def __len__(self):
    # Return the length of the dataset (number of batches)
    # that is given by #images // batch_size
    return len(self.data) // self.batch_size


  def on_epoch_end(self):
    # Shuffle indices after each epoch
    if self.shuffle == True:
        np.random.shuffle(self.indices)

  def get_image_and_label(self, index):

    curr_data = self.data[index] # Get filename at index
    curr_filename = curr_data['path']
    curr_ids = np.array(curr_data['encoding']['input_ids'])
    curr_masks = np.array(curr_data['encoding']['attention_mask'])
    
    image = cv2.imread(curr_filename)

    return image, curr_ids, curr_masks

  def __getitem__(self, index):
    # In this function we generate a batch (of size self.batch_size) of images and corresponding masks
    
    # Get 'self.batch_size' indices
    current_indices = self.indices[index*self.batch_size:(index*self.batch_size)+self.batch_size]

    """if len(current_indices) == 0:
      current_indices = self.indices[len(self.indices)-self.batch_size:len(self.indices)]"""

    # Init lists that will contain images and masks
    batch_images = []
    batch_ids = []
    batch_masks = []

    # Cycle over the indices
    for idx in current_indices:
      # Get single image/mask at index 'idx'
      image, curr_ids, curr_masks = self.get_image_and_label(idx)

      # Apply the preprocessing function
      if self.preprocessing_function is not None:
        if self.channels_first:
          image = np.moveaxis(image, -1, 0)
        image = self.preprocessing_function(image)['pixel_values'][0]
        image = np.moveaxis(image, 0, -1)

      # Append both image and mask (with added batch dimension) to the corresponding batch lists
      batch_images.append(np.expand_dims(image, 0))
      batch_ids.append(curr_ids)
      batch_masks.append(curr_masks)
     
    # Finally, obtain a final batch by concatenating all the images over the batch dimension
    batch_images = tf.convert_to_tensor(np.concatenate(batch_images, axis=0))
    batch_ids = tf.convert_to_tensor(np.array(batch_ids))
    batch_masks = tf.convert_to_tensor(np.array(batch_masks))

    if self.channels_first:
      batch_images = np.moveaxis(batch_images, -1, 1) # in pos 0 there is the batch size

    return (batch_images, batch_ids, batch_masks)
  
##############################################################################################################
# This generator select between a random image and text for each sample, Work In Progress
class ClipUniqueSampleGenerator(ClipBaseGenerator):
  
  def get_image_and_label(self, index):

    curr_data = self.data[index] # Get filename at index
    curr_filename = curr_data['path']
    curr_ids = np.array(curr_data['encoding']['input_ids'])
    curr_masks = np.array(curr_data['encoding']['attention_mask'])
    
    image = cv2.imread(curr_filename)

    return image, curr_ids, curr_masks

  def __getitem__(self, index):
    # In this function we generate a batch (of size self.batch_size) of images and corresponding masks
    
    # Get 'self.batch_size' indices
    current_indices = self.indices[index*self.batch_size:(index*self.batch_size)+self.batch_size]

    """if len(current_indices) == 0:
      current_indices = self.indices[len(self.indices)-self.batch_size:len(self.indices)]"""

    # Init lists that will contain images and masks
    batch_images = []
    batch_ids = []
    batch_masks = []

    # Cycle over the indices
    for idx in current_indices:
      # Get single image/mask at index 'idx'
      image, curr_ids, curr_masks = self.get_image_and_label(idx)

      # Apply the preprocessing function
      if self.preprocessing_function is not None:
        image = self.preprocessing_function(image)

      # Append both image and mask (with added batch dimension) to the corresponding batch lists
      batch_images.append(np.expand_dims(image, 0))
      batch_ids.append(curr_ids)
      batch_masks.append(curr_masks)
     
    # Finally, obtain a final batch by concatenating all the images over the batch dimension
    batch_images = tf.convert_to_tensor(np.concatenate(batch_images, axis=0))
    batch_ids = tf.convert_to_tensor(np.array(batch_ids))
    batch_masks = tf.convert_to_tensor(np.array(batch_masks))

    return (batch_images, batch_ids, batch_masks)
##############################################################################################################



# Utils
def construct_encoding(x, tokenizer,max_len,return_tensors=None):
  return dict(tokenizer(x, max_length=max_len, truncation=True, padding="max_length",return_tensors=return_tensors))


def dup(caption, concepts: set):
  # add caption to concepts set
  concepts.add(caption)
  return concepts

# create new samples, one for each concept and caption to a different sample
def concepts_to_captions_dup(df):
  df = df.copy()
  #copy concepts row to old concepts row
  df['caption_old'] = df['caption']
  # append caption row to concepts set row
  df['caption'] = df[['caption', 'concepts']].apply(lambda x: dup(*x), axis=1)
  # explode concepts set row
  df = df.explode('caption')
  return df

def app(caption, concepts: set):
  # transform concepts in a string separated by commas 
  concepts = ', '.join(concepts)
  return concepts+caption

#Adding concepts at the start of captions separated by commas
def concepts_to_captions_append(df):
  df = df.copy()
  #copy concepts row to old concepts row
  df['caption_old'] = df['caption']
  # append caption row to concepts set row
  df['caption'] = df[['caption', 'concepts']].apply(lambda x: app(*x), axis=1)
  # explode concepts set row
  df = df.explode('caption')
  return df

def remove_stopwords(caption):
  stop_words = set(stopwords.words('english'))
  word_tokens = word_tokenize(caption)
  # converts the words in word_tokens to lower case and then checks whether
  #they are present in stop_words or not
  filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
  #with no lower case conversion
  filtered_sentence = []
  for w in word_tokens:
      if w not in stop_words:
          filtered_sentence.append(w)
  return ' '.join(filtered_sentence)


def remove_words_threshold(caption, threshold = 1):
  word_tokens = word_tokenize(caption)

  filtered_sentence = [w for w in word_tokens if len(w) > threshold]
  #with no lower case conversion

  if len(filtered_sentence) == 0:
     return caption

  return ' '.join(filtered_sentence)

def remove_words(caption, words_to_remove):
  word_tokens = word_tokenize(caption)

  filtered_sentence = [w for w in word_tokens if w not in words_to_remove]
  #with no lower case conversion

  if len(filtered_sentence) == 0:
     return caption

  return ' '.join(filtered_sentence)




#Removes stopwords
def concepts_to_captions_clean(df, words_to_remove=None):
  df = df.copy()
  #copy concepts row to old concepts row
  df['caption_old'] = df['caption']
  nltk.download('stopwords')
  nltk.download('punkt')
  
  # append caption row to concepts set row
  df['caption'] = df['caption'].apply(lambda x: remove_stopwords(x))
  df['caption'] = df['caption'].apply(lambda x: remove_words_threshold(x))
  if words_to_remove:
    df['caption'] = df['caption'].apply(lambda x: remove_words(x, words_to_remove))
  return df

#Removes stopwords and single char words
def concepts_to_captions_remove_stopwords(df):
  df = df.copy()
  #copy concepts row to old concepts row
  df['caption_old'] = df['caption']
  nltk.download('stopwords')
  nltk.download('punkt')
  
  # append caption row to concepts set row
  df['caption'] = df['caption'].apply(lambda x: remove_stopwords(x))
  return df

# Preprocessing function that creates images and labels pairs
def paths_captions_emb_list(df, all_images_path, tokenizer,max_len, remove_images_threshold=None):
    imagesAndLabels = []
    df['caption'] = df['caption'].apply(lambda x:str(x))
    df['ID'] = df['ID'].apply(lambda x:str(x))
    for index, row in tqdm(df.iterrows(),total=df.shape[0]):
        load_image = (remove_images_threshold is None) or (cv2.imread(all_images_path + row.ID, 0).sum() > remove_images_threshold)
        if load_image:
          pair = {
                'path' : all_images_path + row.ID,
                'caption' : row.caption,
                'encoding': construct_encoding(row.caption,tokenizer,max_len)
            }

          imagesAndLabels.append(pair)

    return imagesAndLabels