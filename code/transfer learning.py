########################################
# Overview
#
# Script uses transfer learning to 
# predict for cifar10
#
# Using vgg16 as base model, freeze weights
# and train a dense layer (or maybe a few
# dense layers). Then fine tune.
########################################

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Model

###################
# Load 
###################
IMG_ROWS, IMG_COLS, IMG_CHANNELS = 32, 32, 3
NB_CLASSES = 10

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# reshape
X_train = X_train.reshape((50000, IMG_ROWS, IMG_COLS, IMG_CHANNELS))
X_test = X_test.reshape((10000, IMG_ROWS, IMG_COLS, IMG_CHANNELS))

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

###################
# Define model
###################
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
input_tensor = layers.Input(INPUT_SHAPE)

transfer_model = tf.keras.applications.vgg16.VGG16(input_shape = INPUT_SHAPE, include_top = False, weights = 'imagenet')
transfer_model.trainable = False

base_model = transfer_model(input_tensor)

top_model = layers.Flatten()(base_model)
top_model = layers.Dense(32, activation = 'relu')(top_model)
top_model = layers.Dropout(.30)(top_model)

output_tensor = layers.Dense(NB_CLASSES, activation = 'softmax')(top_model)

model = Model(input_tensor, output_tensor)
model.summary() # Confirm frozen weights
    
###################
# Train model
###################
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.RMSprop(), metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size=512, epochs = 10, verbose = 1, validation_data=(X_test, y_test))

# fine tune
transfer_model.trainable = True
model.summary()
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.RMSprop(lr = .00001), metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size=512, epochs = 7, verbose = 1, validation_data=(X_test, y_test))
