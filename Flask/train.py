# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:58:53 2020

@author: amanm
"""

# Importing the Libraries

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import re
import time
import random
from glob import glob
import pickle 
import collections
import pathlib
import os
from tqdm import tqdm
from model import CNN_Encoder, RNN_Decoder

# Download caption annotation files
annotation_folder = '/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
  annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
  annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
  os.remove(annotation_zip)

# Download image files
image_folder = '/train2014/'
if not os.path.exists(os.path.abspath('.') + image_folder):
  image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                      extract = True)
  PATH = os.path.dirname(image_zip) + image_folder
  os.remove(image_zip)
else:
  PATH = os.path.abspath('.') + image_folder
  
with open("C:/Users/amanm/Desktop/InOut/annotations/captions_train2014.json",'r') as f:
    annotate = json.load(f)
 
# using the image captions from the 
annotations = annotate['annotations']

# create a dic with image path and caption as key value pair
image_path_to_caption = collections.defaultdict(list)

for val in annotations:
    caption = f"<start> {val['caption']} <EOS>"
    image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
    image_path_to_caption[image_path].append(caption)

# shuffle the image paths
image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)

# create two lists to store captions and img_name
train_captions = []
img_name_vector = []

for i in image_paths:
    caption_list = image_path_to_caption[i]
    train_captions.extend(caption_list)
    img_name_vector.extend([i]*len(caption_list))
    
# function to preprocess the image
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img,channels = 3)
    img = tf.image.resize(img, (229,299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, img_path

# import the inception model to extract the initial image features
image_model = tf.keras.applications.InceptionV3(include_top=False, weights= 'imagenet')
new_input = image_model.input
output = image_model.layers[-1].output

image_feature_extract_model = tf.keras.models.Model(new_input,output)

# get the unique image paths

encode_train=sorted(set(img_name_vector))

# make the tensorflow dataset object for better performance

image_dataset=tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset=image_dataset.map(load_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)


''' extract the image features of each image by passing it through the inception model
and save the feautures as a .npz file in the same folder as that of images
'''

for img, path in tqdm(image_dataset):
  batch_features = image_feature_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())

# batch the dataset
image_dataset = image_dataset.batch(16)

# define the vocab size
vocab_size = 20000

# define a function to calculat the max length
def max_len(tensor):
    maxlen = max(len(t) for t in tensor)
    return maxlen

# create the tokenizer instance
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<UNK>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)

# assign pad as the 0th index
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] ='<pad>'

train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs,padding='post')
max_length = max_len(train_seqs)

# create a dict to image to caption as the key value pair
image_to_cap_vector = collections.defaultdict(list)

for image,cap in zip (img_name_vector, cap_vector):
    image_to_cap_vector[image].append(cap)
    
# shuffle the data and split it into train and val

img_keys = list(image_to_cap_vector)
random.shuffle(img_keys)
slice_index = int(len(img_keys)*0.8)

image_name_train_keys, image_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]


# create image name and caption list for both training and validation data
image_name_train = []
cap_train = []

for imgt in tqdm(image_name_train_keys):
    capt_len = len(image_to_cap_vector[imgt])
    image_name_train.extend([imgt]*capt_len)
    cap_train.extend(image_to_cap_vector[imgt])
    
image_name_val=[]
cap_val=[]

for imgv in tqdm(image_name_val_keys):
    capv_len=len(image_to_cap_vector[imgv])
    image_name_val.extend([imgv]*capv_len)
    cap_val.extend(image_to_cap_vector[imgv])
    
    
# Defining some of the global parameters needed for training
Batch_size = 64
BUFFER_SIZE=1000
embedding_dimension=256
units=512
vocab_size=vocab_size+1
num_steps = len(image_name_train) // Batch_size
features_shape=2048
attention_features_shape=64


# extract the features saved in the .npz files
def map_func(image_name,cap):
    image_tensor=np.load(image_name.decode('utf-8')+'.npy')
    return image_tensor,cap

# create a tensorflow dataset and map the above created fuction to each element
dataset=tf.data.Dataset.from_tensor_slices((image_name_train,cap_train))
dataset=dataset.map(lambda item1,item2 :tf.numpy_function(
                    map_func,[item1,item2],[tf.float32,tf.int32]),num_parallel_calls=tf.data.experimental.AUTOTUNE)

# shuffle the data and prefetch it
dataset=dataset.shuffle(BUFFER_SIZE).batch(Batch_size)
dataset=dataset.prefetch(tf.data.experimental.AUTOTUNE)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

encoder = CNN_Encoder(embedding_dimension)
decoder = RNN_Decoder(embedding_dimension, units, vocab_size)


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []


start_epoch = 12
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)
  
@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss
  
# train the model on gpu
EPOCHS = 20
with tf.device('GPU'):
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                  epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        
        ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                             total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
