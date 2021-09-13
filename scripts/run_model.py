# This script is used to run trained models
# It comes handy when you want to debug, test generated models by AutoDaedalus

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from deepswarm import anomalies
import sys
from vizualization import painter
from vizualization.painter import reconstructed_results

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Autoencoder's location on disk
autoencoder_path = '../specify/path/on/disk'

# Encoder's location on disk (used for function: painter.encoded_image())
encoder_path = path +'/encoder_model'

sys.stdout = open(path + '/run_log.txt', 'w')

# Lists of valid labels and anomalies
valid_label = [1,2,3,4,5,6,7,8,9]
anomaly_label = [0]

# Load autoencoder model from disk
autoencoder = keras.models.load_model(autoencoder_path)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load encoder model from disk
encoder = keras.models.load_model(encoder_path)
encoder.compile()

# Build a testing dataset
((trainX, trainY), (testX, testY)) = tf.keras.datasets.mnist.load_data()

validIdxs = list()
for i in range(len(testY)):
    if testY[i] in list(valid_label):
        validIdxs.append(i)

anomalyIdxs = list()
for i in range(len(testY)):
    if testY[i] in list(anomaly_label):
        anomalyIdxs.append(i)

validImages = testX[validIdxs]
validImages_labels = testY[validIdxs]
anomalyImages = testX[anomalyIdxs]
anomalyImages_labels = testY[anomalyIdxs]

images = np.vstack([validImages, anomalyImages])
labels = list()
labels = list(validImages_labels) + list(anomalyImages_labels)

images = np.expand_dims(images, axis=-1)
images = images.astype("float32") / 255.0

testX = images
testY = labels

plt_recunstructed_results = reconstructed_results(autoencoder,testX)
plt_recunstructed_results.savefig(f'{path}/plt_recunstructed_results.png')
plt_recunstructed_results.show()

plt_encoded_image = painter.encoded_image(autoencoder, encoder, testX)
plt_encoded_image.savefig(f'{path}/plt_encoded_image.png')
plt_encoded_image.show()

# Find anomalies in data (multiple quantiles)
print(f"=====================================")
print(f"Finding anomalies in quantile: 0.995")
plt_anomalies = anomalies.find(autoencoder, testX, testY, 0.995, True, valid_label, anomaly_label)
plt_anomalies.savefig(f'{path}/plt_anomalies_0{str(995)}.png')
plt_anomalies.show()

print(f"=====================================")
print(f"Finding anomalies in quantile: 0.98")
plt_anomalies = anomalies.find(autoencoder, testX, testY, 0.98, True, valid_label, anomaly_label)
plt_anomalies.savefig(f'{path}/plt_anomalies_0{str(98)}.png')
plt_anomalies.show()

print(f"=====================================")
print(f"Finding anomalies in quantile: 0.9")
plt_anomalies = anomalies.find(autoencoder, testX, testY, 0.9, True, valid_label, anomaly_label)
plt_anomalies.savefig(f'{path}/plt_anomalies_0{str(9)}.png')
plt_anomalies.show()

# Evaluate model
roc_curve = anomalies.calculate_roc_curve(autoencoder, testX, testY, True)
roc_curve.savefig(f'{path}/roc_curve.png', dpi=300)
roc_curve.show()

sys.stdout.close()
