import seaborn
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from matplotlib import pyplot as plt

from deepswarm import cfg


def training_loss(history, cfg, epochs=cfg['backend']['epochs']):
    # Loss
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")

    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    return plt


def training_acc(history, cfg, epochs=cfg['backend']['epochs']):
    # Accuracy
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()

    plt.plot(N, history.history['accuracy'], label="train_acc")
    plt.plot(N, history.history['val_accuracy'], label="val_acc")

    plt.title("Training accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    return plt


def reconstructed_results(model, x_test, storage):
    """Visualize on image predicted results and train data
    Args:
        model: model which represents neural network structure.
        x_test: desired output image
    """

    decoded_imgs = model.predict(x_test)
    vol_size = K.int_shape(x_test)
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

    return plt


def MAE_loss(model, x_train, storage):
    # Get train MAE loss.
    decoded = model.predict(x_train)
    errors = []

    # loop over all original images and their corresponding
    # reconstructions
    for (image, recon) in zip(x_train, decoded):
        # compute the mean squared error between the ground-truth image
        # and the reconstructed image, then add it to our list of errors
        mse = np.mean((image - recon) ** 2)
        errors.append(mse)

    plt.hist(errors, bins=50)
    plt.xlabel("Train MAE loss")
    plt.ylabel("No of samples")

    # Get reconstruction loss threshold.
    threshold = np.max(errors)
    print("Reconstruction error threshold: ", threshold)
    return plt


def encoded_image(model, x_test):
    # https://stackoverflow.com/questions/51091106/correct-way-to-get-output-of-intermediate-layer-in-keras-model
    layer_name = 'encoder'
    intermediate_layer_model = keras.Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x_test)

    fig, ax = plt.subplots()
    seaborn.heatmap([intermediate_output[0]])
    return plt
