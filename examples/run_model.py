import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from deepswarm import anomalies

from vizualization import painter
from vizualization.painter import reconstructed_results

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#path = '/home/spartan/PycharmProjects/DeepSwarm/saves/2021-08-21-00-13-37_multi_label/best_anomaly_detector_depth_5/4eeb69229bf7051a34c8526dc1fe77e2a6804da917bd989d179caf0e924b8d0e'
path = '/home/spartan/PycharmProjects/DeepSwarm/saves/2021-08-20-09-47-39_single_label_1V_0A/models/6917d7c05d4c5590cf3a537ee6ce1333019d61a5a3b1d98ec1117b1074e47ecd'

autoencoder_path = path + '/'
encoder_path = path +'/encoder_model'

# path = '../saves/2021-08-19-23-50-24_single_label_16_ants/models/'
# autoencoder_path = path + 'best-trained-topology'
# encoder_path = path +'3a4cc9cd7f6a070666407d1241b82d8c1b82329e01a181b098ff060e9ef26d3e/encoder_model'

valid_label = [1]
anomaly_label = [0]

autoencoder = keras.models.load_model(autoencoder_path)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

encoder = keras.models.load_model(encoder_path)
encoder.compile()

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

predicted = autoencoder.predict(testX)
predicted_labels = labels


# evaluate the model
loss, accuracy = autoencoder.evaluate(testX, predicted, verbose=0)

plt_recunstructed_results = reconstructed_results(autoencoder,testX)
plt_recunstructed_results.show()

plt_encoded_image = painter.encoded_image(autoencoder, encoder, testX)
plt_encoded_image.show()

# Find anomalies in data
plt_anomalies = anomalies.find(autoencoder, testX, testY, 0.95, True, valid_label, anomaly_label)
plt_anomalies.show()

# Evaluate model
roc_curve = anomalies.calculate_roc_curve(autoencoder, testX, testY, True)
roc_curve.show()