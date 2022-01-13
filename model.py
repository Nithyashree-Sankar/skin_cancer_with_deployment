import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
#from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import random 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# re-size all the images to standard size 224 x 224
IMAGE_SIZE = [224, 224]

# specifying path for train and test data folders
train_path = '../input/skin-cancer-malignant-vs-benign/train'
valid_path = '../input/skin-cancer-malignant-vs-benign/test'
#function to visualize image
def plot_image(file, directory=None, sub=False, aspect=None):
    path = directory + file
    
    img = plt.imread(path)
    
    plt.imshow(img, aspect=aspect)
#     plt.title(file)
    plt.xticks([])
    plt.yticks([])
    
    if sub:
        plt.show()
def plot_img_dir(directory=train_path, count=5):
    selected_files = random.sample(os.listdir(directory), count)
    
    ncols = 5
    nrows = count//ncols if count%ncols==0 else count//ncols+1
    
    figsize=(20, ncols*nrows)

    ticksize = 14
    titlesize = ticksize + 8
    labelsize = ticksize + 5


    params = {'figure.figsize' : figsize,
              'axes.labelsize' : labelsize,
              'axes.titlesize' : titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}

    plt.rcParams.update(params)
    
    i=0
    
    for file in selected_files:        
        plt.subplot(nrows, ncols, i+1)
        path = directory + file
        plot_image(file, directory, aspect=None)

        i=i+1
    
    plt.tight_layout()
    plt.show()
    
def plot_img_dir_main(directory=train_path, count=5):
    labels = os.listdir(directory)
    for label in labels:
        print(label)
        plot_img_dir(directory=directory+"/"+label+"/", count=count)
plot_img_dir_main(directory=train_path, count=5)
# added preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# Freezing the existing layers 
for layer in vgg.layers:
  layer.trainable = False

# fetching number of classes
folders = glob('../input/skin-cancer-malignant-vs-benign/train/*')
# Flatten the output layer to 1 dimension
#x = layers.Flatten()(base_model.output)
x = Flatten()(vgg.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = Dropout(0.5)(x)

# Add a fully connected layer with 256 hidden units and ReLU activation
x = Dense(256, activation='relu')(x)

# Add a dropout rate of 0.5
x = Dropout(0.5)(x)

# Add a final sigmoid layer for classification
prediction = Dense(len(folders), activation='sigmoid')(x)

#model = tf.keras.models.Model(base_model.input, x)

#model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
# model object creation
model = Model(inputs=vgg.input, outputs=prediction)
# view model structure
model.summary()
# setting model cost and optimization method to use
model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
# Image Data Generator to import the images from the dataset
from keras.preprocessing.image import ImageDataGenerator
# Addition of data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   rotation_range = 40,
                                   horizontal_flip = True)
 # validation data should not be augmented
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../input/skin-cancer-malignant-vs-benign/train',
                                                 target_size = (224, 224),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('../input/skin-cancer-malignant-vs-benign/test',
                                            target_size = (224, 224),
                                            batch_size = 64,
                                            class_mode = 'categorical')
# number of  benign test data
DIR = '../input/skin-cancer-malignant-vs-benign/test/benign'
benign_test = ([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
print("number of benign test data:" + str(len(benign_test)))
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=100,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks = [learning_rate_reduction]
)
X_test, y_test = next(test_set)
print("Loss of the model is - " , model.evaluate(X_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(X_test,y_test)[1]*100 , "%")
import tensorflow as tf

from keras.models import load_model

# save the fine tuned model
model.save('model_vgg19.h5')
model.save('model_pb')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
#model=load_model('model_vgg19.h5')../input/modellvgg19
model=load_model('model_vgg19.h5')
img=image.load_img('../input/skin-cancer-malignant-vs-benign/train/malignant/100.jpg',target_size=(224,224))

x=image.img_to_array(img)
x=x/255
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
model.predict(img_data)
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
a=np.argmax(model.predict(img_data), axis=1)
if(a==0):
    print("Uninfected")
else:
    print("Infected")
