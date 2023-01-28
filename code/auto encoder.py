#####################################################
# Overview
#
# A simple small scale auto encoder.
#####################################################

# %%
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import make_classification
import numpy as np


# %%
X, _ = make_classification(n_samples=500000, n_features = 1000, n_informative=1000, n_redundant=0, n_repeated=0, random_state=42)

# %%
input_tensor = tf.keras.Input(shape = (X.shape[1], ))
x = layers.Dense(128, activation = 'linear')(input_tensor)
x = layers.Dense(64, activation = 'relu')(x)
x = layers.Dense(128, activation = 'relu')(x)
output_tensor = layers.Dense(X.shape[1], activation = 'linear')(x)

model = tf.keras.Model(input_tensor, output_tensor)
model.compile(loss =tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam())
model.summary()

# %%
model.fit(x = X, y = X, epochs = 10, batch_size = 2**10)

# %%
model.evaluate(x = X,y = X, batch_size = 2**10)