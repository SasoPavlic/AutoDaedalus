import datetime
import os

import numpy
import seaborn
from keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.utils import *
from matplotlib import pyplot as plt



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
plot_model(encoder_model, to_file='./logs/encoder_plot.png', show_shapes=True, show_layer_names=True)


latentInputs = Input(shape=(32,))
decoded = Dense(np.prod(volumeSize[1:]))(latentInputs)
decoded = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(decoded)
decoded = Dense(128, activation='relu')(decoded)
encoded = Dense(256, activation='relu')(encoded)
decoded = Dense(3, activation='sigmoid')(decoded)
decoder_model = tf.keras.Model(inputs=latentInputs, outputs=decoded, name='decoder')


plot_model(decoder_model, to_file='./logs/decoder_plot.png', show_shapes=True, show_layer_names=True)

autoencoder = keras.Model(input_img, decoder_model(encoder_model(input_img)),name="autoencoder")
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC'])

print(autoencoder.summary())
plot_model(autoencoder, to_file='./logs/autoencoder_plot.png', show_shapes=True, show_layer_names=True)

from tensorflow.keras.datasets import cifar100
import numpy as np

(x_train, _), (x_test, _) = cifar100.load_data(label_mode="fine")

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))



log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#autoencoder.layers[0].trainable = False
# tensorboard --logdir ./logs/fit/
autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tensorboard_callback])

# https://stackoverflow.com/questions/51091106/correct-way-to-get-output-of-intermediate-layer-in-keras-model
layer_name = 'encoder'
intermediate_layer_model = keras.Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(x_test)


decoded_imgs = autoencoder.predict(x_test)
decoded_imgs = intermediate_output
score, acc, *all_metrices = autoencoder.evaluate(x_test, x_test, batch_size=128)




test= K.int_shape(x_test)
print()
print('Score: %1.4f' % score)
print('Evaluation Accuracy: %1.2f%%' % (acc*100))

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
