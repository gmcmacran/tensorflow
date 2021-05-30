####################################
# Overview
#
# An example of a deep learning
# regression model with a residual 
# connection.
####################################

from tensorflow import keras 
from tensorflow.keras import layers
from sklearn.datasets import make_s_curve
from sklearn.metrics import mean_absolute_error

##################
# Make data
##################
X_train, y_train = make_s_curve(100000, random_state=1)
X_test, y_test = make_s_curve(100000, random_state=2)


##################
# Model
##################
input_tensor = keras.Input(shape=(X_train.shape[1],))
x = layers.Dense(10, activation = 'relu')(input_tensor)
y = layers.Dense(10, activation = 'relu')(x)
y = layers.Dense(10, activation = 'relu')(y)
y = layers.Dense(10, activation = 'relu')(y)
both = layers.add([x, y])
output_tensor = layers.Dense(1, activation = 'linear')(both)

model = keras.Model(input_tensor, output_tensor)
model.summary()

model.compile(loss =keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam())
model.fit(X_train, y_train, epochs = 10, batch_size=256)

##################
# Metrics
##################
mean_absolute_error(y_train, model.predict(X_train))
mean_absolute_error(y_test, model.predict(X_test))
