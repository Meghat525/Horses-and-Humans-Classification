#Downloading the dataset

import urllib.request
urllib.request.urlretrieve(" https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip", "horse-or-human.zip")
import urllib.request
urllib.request.urlretrieve("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip ", "validation-horse-or-human.zip")

import zipfile
local_zip = 'horse-or-human-spyder.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('horse-or-human')
local_zip = 'validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('validation-horse-or-human')
zip_ref.close()

import os #operating system
# Directory with our training horse pictures
train_horse_dir = os.path.join('horse-or-human/horses')
#The folder "horse-or-human is stored in the jupyter notebook"
# Directory with our training human pictures
train_human_dir = os.path.join('horse-or-human/humans')

# Directory with our testing horse pictures
validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')
# Directory with our testing human pictures
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')

#looking at the filenames
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

#total number of horse and human images
print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))

#look at a few pictures
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) 
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) 
                for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255) #Normalizing
train_generator=train_datagen.flow_from_directory('horse-or-human',
                                                 target_size=(300,300),
                                                 batch_size=128,
                                                 class_mode="binary")
validation_generator=train_datagen.flow_from_directory("validation-horse-or-human",
                                                 target_size=(300,300),
                                                 batch_size=32,
                                                 class_mode="binary")
model = tf.keras.models.Sequential([#first convolution
                                    keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation="relu",input_shape=(300,300,3)),
                                    keras.layers.MaxPooling2D(2,2),
                                    #second convolution
                                    keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu"),
                                    keras.layers.MaxPooling2D(2,2),
                                    #third convolution
                                    keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"),
                                    keras.layers.MaxPooling2D(2,2),
                                    # The fourth convolution
                                    keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    keras.layers.MaxPooling2D(2,2),
                                    # The fifth convolution
                                    keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    keras.layers.MaxPooling2D(2,2),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(512,activation=tf.nn.relu),
                                    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
                                    keras.layers.Dense(1,activation=tf.nn.sigmoid)])
model.compile(loss=tf.losses.BinaryCrossentropy(),optimizer=tf.optimizers.RMSprop(lr=0.001),metrics=["accuracy"])
model.fit_generator(train_generator,
                   steps_per_epoch=8,
                   epochs=15,
                   validation_data=validation_generator,
                   validation_steps=8,
                   verbose=1) #total no. of images=1024, batch_size=128. So, steps_per_epoch=1024/128=8
