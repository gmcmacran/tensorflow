#########################################
# Overview
# 
# Building a GAN to create images of 
# numbers that look man made but are actually
# computer generated.
#
# Based on Deep Learning with Tensorflow 2
# and Keras book.
#
# Convergence behavior changes run to run.
# Sometimes the generated images are great.
# Sometimes they are not.
#
# With 4070 TI, it takes about 10 min for
# 5,000 epochs.
#
# A modification of 
# https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-Keras/blob/master/Chapter%206/DCGAN.ipynb
#########################################

# %%
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from plotnine import ggplot, aes, ggsave
from plotnine import geom_line, geom_hline
from plotnine import scale_x_continuous, scale_y_continuous


# %%
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

latent_dim = 30

# %%
def build_generator():

    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    # model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

# %%
def build_discriminator():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


# %%
dLossesFake = []
dLossesReal = []
gLosses = []
accs = []
def train(epochs, batch_size=256, save_interval=50):

    def save_imgs(epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fn = "s:\\Python\\projects\\tensorflow\\images\\dcgan_mnist_%d.png" % epoch
        fig.savefig(fn)
        plt.close()

    optimizer = Adam(0.0002, 0.5)
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    generator = build_generator()

    discriminator.trainable = False

    z = Input(shape=(latent_dim,))
    img = generator(z)
    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    if batch_size > X_train.shape[0]:
        batch_size = X_train.shape[0]

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    prevDLoss = 100 # holder value
    prevGLoss = 100 # holder value
    prevAcc = 100

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        # Train the discriminator (real classified as ones and generated as zeros)
        if (.9 * prevDLoss >= prevGLoss and prevAcc < .30 and prevDLoss > .20):
            discriminator.trainable = True
        else:
            discriminator.trainable = False
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (wants discriminator to mistake images as real)
        discriminator.trainable = False
        g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # Store loss of most recent batch from this epoch
        dLossesReal.append(d_loss_real[0])
        dLossesFake.append(d_loss_fake[0])
        gLosses.append(g_loss)
        accs.append(d_loss[1])

        # save for next iteration
        prevDLoss = d_loss[0] 
        prevGLoss = g_loss
        prevAcc = d_loss[1] 

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_imgs(epoch)

    return(generator)

# %%
tic = time.perf_counter()
train(epochs=5000, batch_size=1024, save_interval=200)
toc = time.perf_counter()
print(f"Training time: {(toc - tic)/60:0.0f} minutes and {(toc - tic)%60:0.0f} seconds")

# %%
epochs = list(range(len(dLossesFake)))
data = {'Epoch': epochs, 'FakeDLoss': dLossesFake, 'RealDLoss': dLossesReal, 'GLoss': gLosses}
DF = pd.DataFrame(data = data)
DF = DF.melt(id_vars='Epoch', value_vars=['FakeDLoss', 'RealDLoss', 'GLoss'])

graph = (
    ggplot(DF, aes(x = 'Epoch', y = 'value', color = 'variable', group = 'variable'))
    + geom_line()
    + scale_x_continuous(breaks = range(0, len(epochs), 400))
    + geom_hline(yintercept = .15)
)
fn = "s:\\Python\\projects\\tensorflow\\images\\loss_functions.png"
ggsave(plot = graph, filename = fn, width = 10, height = 10)
graph

# %%
epochs = list(range(len(accs)))
data = {'Epoch': epochs, 'Accuracy': accs}
DF = pd.DataFrame(data = data)

graph = (
    ggplot(DF, aes(x = 'Epoch', y = 'Accuracy'))
    + geom_line()
    + scale_x_continuous(breaks = range(0, len(epochs), 400))
    + scale_y_continuous(breaks = np.arange(0.0, 1.1, .10), limits = [0.0, 1.0])
)
fn = "s:\\Python\\projects\\tensorflow\\images\\accuracy.png"
ggsave(plot = graph, filename = fn, width = 10, height = 10)
graph


# %%
lossGraphs = []
accGraphs = []
for i in range(0, 10, 1):
    dLossesFake = []
    dLossesReal = []
    gLosses = []
    accs = []

    train(epochs=5001, batch_size=1024, save_interval=250)

    epochs = list(range(len(dLossesFake)))
    data = {'Epoch': epochs, 'FakeDLoss': dLossesFake, 'RealDLoss': dLossesReal, 'GLoss': gLosses}
    DF = pd.DataFrame(data = data)
    DF = DF.melt(id_vars='Epoch', value_vars=['FakeDLoss', 'RealDLoss', 'GLoss'])

    graph = (
        ggplot(DF, aes(x = 'Epoch', y = 'value', color = 'variable', group = 'variable'))
        + geom_line()
        + scale_x_continuous(breaks = range(0, len(epochs), 500))
        + geom_hline(yintercept = .20)
    )
    fn = "s:\\Python\\projects\\tensorflow\\images\\loss_functions.png"
    ggsave(plot = graph, filename = fn, width = 10, height = 10)
    lossGraphs.append(graph)

    epochs = list(range(len(accs)))
    data = {'Epoch': epochs, 'Accuracy': accs}
    DF = pd.DataFrame(data = data)

    graph = (
        ggplot(DF, aes(x = 'Epoch', y = 'Accuracy'))
        + geom_line()
        + scale_x_continuous(breaks = range(0, len(epochs), 500))
        + scale_y_continuous(breaks = np.arange(0.0, 1.1, .10), limits = [0.0, 1.0])
    )
    fn = "s:\\Python\\projects\\tensorflow\\images\\accuracy.png"
    ggsave(plot = graph, filename = fn, width = 10, height = 10)
    accGraphs.append(graph)

# %%
for i in range(0, 1, 1):
    lossGraphs[i]

# %%
for i in range(1):
    accGraphs[i]