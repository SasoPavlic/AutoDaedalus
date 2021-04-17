from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
import numpy as np
import cv2


def visualize_predictions(decoded, gt, samples=10):
    # initialize our list of output images
    outputs = None

    # loop over our number of output samples
    for i in range(0, samples):
        # grab the original image and reconstructed image
        original = (gt[i] * 255).astype("uint8")
        recon = (decoded[i] * 255).astype("uint8")

        # stack the original and reconstructed image side-by-side
        output = np.hstack([original, recon])

        # if the outputs array is empty, initialize it as the current
        # side-by-side image display
        if outputs is None:
            outputs = output

        # otherwise, vertically stack the outputs
        else:
            outputs = np.vstack([outputs, output])

    # return the output images
    return outputs


def show_training_graph(model, x_test, history, cfg):
    decoded_imgs = model.predict(x_test)
    vis = visualize_predictions(decoded_imgs, x_test)
    cv2.imshow("Predictions", vis)

    # construct a plot that plots and saves the training history
    N = np.arange(0, cfg['backend']['epochs'])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.show()


def show_results_on_figure(model, x_test):
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


def anomaly(model, x_test, cfg):
    decoded = model.predict(x_test)
    errors = []

    # loop over all original images and their corresponding
    # reconstructions
    for (image, recon) in zip(x_test, decoded):
        # compute the mean squared error between the ground-truth image
        # and the reconstructed image, then add it to our list of errors
        mse = np.mean((image - recon) ** 2)
        errors.append(mse)

    # compute the q-th quantile of the errors which serves as our
    # threshold to identify anomalies -- any data point that our model
    # reconstructed with > threshold error will be marked as an outlier
    thresh = np.quantile(errors, cfg['aco']['quantile'])
    idxs = np.where(np.array(errors) >= thresh)[0]
    print("[INFO] mse threshold: {}".format(thresh))
    print("[INFO] {} outliers found".format(len(idxs)))

    # initialize the outputs array
    outputs = None

    # loop over the indexes of images with a high mean squared error term
    for i in idxs:
        # grab the original image and reconstructed image
        original = (x_test[i] * 255).astype("uint8")
        recon = (decoded[i] * 255).astype("uint8")

        # stack the original and reconstructed image side-by-side
        output = np.hstack([original, recon])

        # if the outputs array is empty, initialize it as the current
        # side-by-side image display

        if outputs is None:
            outputs = output

        # otherwise, vertically stack the outputs
        else:
            outputs = np.vstack([outputs, output])

    # show the output visualization
    vol_size = K.int_shape(outputs)
    plt.imshow(outputs.reshape(vol_size[0], vol_size[1]))
    plt.show()
