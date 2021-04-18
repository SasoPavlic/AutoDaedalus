from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

layers = list()

input_img = Input(shape=(32, 32, 3))
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
volumeSize = K.int_shape(encoded)
flatten = Flatten()(encoded)
flatten = Dense(32)(flatten)
encoder_model = tf.keras.Model(inputs=input_img, outputs=flatten, name='encoder')

latentInputs = Input(shape=(32,))
decoded = Dense(np.prod(volumeSize[1:]))(latentInputs)
decoded = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(decoded)
decoded = Dense(128, activation='relu')(decoded)
encoded = Dense(256, activation='relu')(encoded)
decoded = Dense(3, activation='sigmoid')(decoded)
decoder_model = tf.keras.Model(inputs=latentInputs, outputs=decoded, name='decoder')

print(decoder_model.summary())

autoencoder = keras.Model(input_img, decoder_model(encoder_model(input_img)),name="autoencoder")
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#autoencoder.summary()

from tensorflow.keras.datasets import cifar100
import numpy as np

(x_train, _), (x_test, _) = cifar100.load_data(label_mode="fine")

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

from tensorflow.keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

decoded_imgs = autoencoder.predict(x_test)
score, acc = autoencoder.evaluate(x_test, x_test, batch_size=128)

test= K.int_shape(x_test)
print()
print('Score: %1.4f' % score)
print('Evaluation Accuracy: %1.2f%%' % (acc*100))

from matplotlib import pyplot as plt
vol_size= K.int_shape(x_test)
dims = (None, None, None)
if vol_size[3] > 1:
    dims = (vol_size[1], vol_size[2], vol_size[3])
else:
    dims = (vol_size[1], vol_size[2])

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(dims))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(dims))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
