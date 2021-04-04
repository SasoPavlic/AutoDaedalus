from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

layers = list()
#ENCODER
inp = Input((28, 28,1))
layers.append(inp)
e = Conv2D(32, (3, 3), activation='relu')(inp)
layers.append(e)
e = MaxPooling2D((2, 2))(e)
layers.append(e)
e = Conv2D(64, (3, 3), activation='relu')(e)
layers.append(e)
e = MaxPooling2D((2, 2))(e)
layers.append(e)
e = Conv2D(64, (3, 3), activation='relu')(e)
layers.append(e)
l = Flatten()(e)
layers.append(l)
l = Dense(49, activation='softmax')(l)
layers.append(l)

#DECODER
d = Reshape((7,7,1))(l)
layers.append(d)
d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)
layers.append(d)
d = BatchNormalization()(d)
layers.append(d)
d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)
layers.append(d)
d = BatchNormalization()(d)
layers.append(d)
d = Conv2DTranspose(32,(3, 3), activation='relu', padding='same')(d)
layers.append(d)
decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(d)
layers.append(decoded)
autoencoder = keras.Model(inp, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()

from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

from tensorflow.keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=3,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

decoded_imgs = autoencoder.predict(x_test)
score, acc = autoencoder.evaluate(x_test, x_test, batch_size=128)

print()
print('Score: %1.4f' % score)
print('Evaluation Accuracy: %1.2f%%' % (acc*100))

from matplotlib import pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()