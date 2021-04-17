# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import context
import os
import tensorflow as tf
import numpy as np
from deepswarm.backends import Dataset, TFKerasBackend
from deepswarm.deepswarm import DeepSwarm
from deepswarm.dataset import prepare_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

(x_train, x_test) = prepare_dataset()

# # Load MNIST dataset
# mnist = tf.keras.datasets.mnist
# (x_train, _), (x_test, _) = mnist.load_data()
# # Normalize and reshape data
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Create dataset object, which controls all the data
normalized_dataset = Dataset(
    training_examples=x_train,
    training_labels=x_train,
    testing_examples=x_test,
    testing_labels=x_test
)
# Create backend responsible for training & validating
backend = TFKerasBackend(dataset=normalized_dataset)
# Create DeepSwarm object responsible for optimization
deepswarm = DeepSwarm(backend=backend)
# Find the topology for a given dataset
topology = deepswarm.find_topology()
# Evaluate discovered topology
deepswarm.evaluate_topology(topology)
# Train topology for additional 30 epochs
trained_topology = deepswarm.train_topology(topology, 30)
# Evaluate the final topology
deepswarm.evaluate_topology(trained_topology)
