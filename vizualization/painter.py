from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
import numpy as np
from deepswarm import cfg

def show_training_graph(history, cfg, epochs=cfg['backend']['epochs']):
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
    plt.show()

    # Accuracy
    plt.style.use("ggplot")
    plt.figure()

    plt.plot(N, history.history['accuracy'], label="train_acc")
    plt.plot(N, history.history['val_accuracy'], label="val_acc")

    plt.title("Training accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.show()


def show_results_on_figure(model, x_test, storage):
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
    plt.show()


def show_MAE_loss(model, x_train, storage):
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
    plt.show()

    # Get reconstruction loss threshold.
    threshold = np.max(errors)
    print("Reconstruction error threshold: ", threshold)
