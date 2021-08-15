from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import random
import tensorflow as tf
from .log import Log
import matplotlib.pyplot as plt


def build_unsupervised_dataset(data, labels, validLabel,
	anomalyLabel, contamination, seed=42):
	# grab all indexes of the supplied class label that are *truly*
	# that particular label, then grab the indexes of the image
	# labels that will serve as our "anomalies"

	validIdxs = np.where(labels == validLabel)[0]
	anomalyIdxs = np.where(labels == anomalyLabel)[0]

	# randomly shuffle both sets of indexes
	random.shuffle(validIdxs)
	random.shuffle(anomalyIdxs)

	# compute the total number of anomaly data points to select
	i = int(len(validIdxs) * contamination)
	anomalyIdxs = anomalyIdxs[:i]

	# use NumPy array indexing to extract both the valid images and
	# "anomlay" images
	validImages = data[validIdxs]
	anomalyImages = data[anomalyIdxs]

	# stack the valid images and anomaly images together to form a
	# single data matrix and then shuffle the rows
	images = np.vstack([validImages, anomalyImages])
	np.random.seed(seed)
	np.random.shuffle(images)
	# return the set of images
	return images


def prepare_dataset(validLabel, anomalyLabel, contamination, test_size, random_state):
	# load the MNIST dataset
	print("[INFO] loading MNIST dataset...")
	# DATASET
	((trainX, trainY), (testX, testY)) = tf.keras.datasets.mnist.load_data()

	# build our unsupervised dataset of images with a small amount of
	# contamination (i.e., anomalies) added into it
	print("[INFO] creating unsupervised dataset...")
	images = build_unsupervised_dataset(trainX, trainY, validLabel, anomalyLabel, contamination)

	# add a channel dimension to every image in the dataset, then scale
	# the pixel intensities to the range [0, 1]
	images = np.expand_dims(images, axis=-1)
	images = images.astype("float32") / 255.0

	# construct the training and testing split
	(trainX, testX) = train_test_split(images, test_size=test_size, random_state=random_state)

	return (trainX, testX)


