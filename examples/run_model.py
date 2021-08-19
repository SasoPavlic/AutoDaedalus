import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf

from deepswarm import anomalies
from deepswarm.dataset import prepare_dataset
from vizualization import painter
from vizualization.painter import reconstructed_results

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

autoencoder_path = '../saves/2021-08-18-16-25-12_single_label_8_ants/models/best-trained-topology'
encoder_path = '../saves/2021-08-18-16-25-12_single_label_8_ants/models/dbabcd362317ae8619615c87b8ab040e937aa19d9848f0bba7bdfb7c48573a91/encoder_model'
valid_label = [7]
anomaly_label = [1]
# path = '../saves/2021-08-18-13-16-51_multi_label_8_ants/models/'
# autoencoder_path = path + 'best-trained-topology'
# encoder_path = path +'2815f29f413784e6d9a4e19009315fd4d465ccf12628703dcd8c396ed05530c2/encoder_model'

# valid_label = [1,2,3,4,5,6,7,8,9]
# anomaly_label = [0]

autoencoder = keras.models.load_model(autoencoder_path)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

encoder = keras.models.load_model(encoder_path)
encoder.compile()

((trainX, trainY), (testX, testY)) = tf.keras.datasets.mnist.load_data()

#validIdxs = np.where(trainY == valid_label)[0]
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

X_train, testX, y_train, testY = train_test_split(images,labels, test_size=0.2, random_state=42)



plt_recunstructed_results = reconstructed_results(autoencoder,testX)
plt_recunstructed_results.show()

plt_encoded_image = painter.encoded_image(autoencoder, encoder, testX)

plt_encoded_image.show()

# Find anomalies in data
plt_anomalies = anomalies.find(autoencoder, testX, testY, 0.98, True, valid_label, anomaly_label)
plt_anomalies.show()
