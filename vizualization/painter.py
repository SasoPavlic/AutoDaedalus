import numpy
import seaborn
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
from deepswarm.log import Log

from deepswarm import cfg

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def training_loss(history, epochs=cfg['backend']['epochs']):
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


def training_acc(history, epochs=cfg['backend']['epochs']):
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


def reconstructed_results(model, x_test):
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

    (x_test, decoded_imgs) = unison_shuffled_copies(x_test, decoded_imgs)
    n = 20
    plt.figure(figsize=(40, 4))
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


def MAE_loss(model, x_train, manual_code=False):
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

    if manual_code:
        print(f"Reconstruction error threshold: {threshold}")
    else:
        Log.info(f"Reconstruction error threshold: {threshold}")
    return plt



def encoded_image(autoencoder, encoder, x_test):
    encoded_imgs = encoder.predict(x_test)
    predicted = autoencoder.predict(x_test)

    p = numpy.random.permutation(len(x_test))
    (x_test, encoded_imgs, predicted) = (x_test[p], encoded_imgs[p], predicted[p])
    # Display 40 images
    plt.figure(figsize=(40, 4))
    for i in range(20):
        # display original images
        ax = plt.subplot(3, 20, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display encoded images
        ax = plt.subplot(3, 20, i + 1 + 20)
        plt.imshow(encoded_imgs[i].reshape(4, 4))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstructed images
        ax = plt.subplot(3, 20, 2 * 20 + i + 1)
        plt.imshow(predicted[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    return plt
