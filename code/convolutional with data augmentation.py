########################################
# Overview
#
# Script trains a convolutional neural 
# network using data augmentation.
#
# Takes a while to execute on a Nvidia 1070.
########################################

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

##################
# Network structure
##################
EPOCHS = 15
BATCH_SIZE = 512
VERBOSE = 1
VALIDATION_SPLIT = .10

IMG_ROWS, IMG_COLS, IMG_CHANNELS = 32, 32, 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
NB_CLASSES = 10

def build(input_shape, classes):
    model = models.Sequential()
    
    model.add(layers.Conv2D(20, (5, 5), activation = 'relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    model.add(layers.Conv2D(50, (5, 5), activation = 'relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation = 'relu'))
    
    model.add(layers.Dense(classes, activation='softmax'))
    
    return model

##################
# Shape data 
##################
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

X_train = X_train.reshape((50000, IMG_ROWS, IMG_COLS, IMG_CHANNELS))
X_test = X_test.reshape((10000, IMG_ROWS, IMG_COLS, IMG_CHANNELS))

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

##################
# data augmentation
##################
datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

##################
# Train model
##################
model = build(INPUT_SHAPE, NB_CLASSES)
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.RMSprop(), metrics = ['accuracy'])

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE ), epochs = EPOCHS, verbose = VERBOSE, validation_data=(X_test, y_test))
