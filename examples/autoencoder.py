# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import os
from deepswarm.backends import Dataset, TFKerasBackend
from deepswarm.deepswarm import DeepSwarm
from deepswarm.dataset import prepare_dataset
from deepswarm import data_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Contaminate dataset with anomalies (e.g. dataset with 99% of 1 digits and 1% of 3 digits)
(x_train, x_test) = prepare_dataset(validLabel=data_config["valid_label"],
                                    anomalyLabel=data_config["anomaly_label"],
                                    contam=data_config["contamination"])

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
