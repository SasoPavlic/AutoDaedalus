from matplotlib import pyplot as plt
from tensorflow.keras import backend as K


def show_results_on_figure(model, x_test):
    """Visualize on image predicted results and train data

    Args:
        model: model which represents neural network structure.
        x_test: desired output image
    """

    decoded_imgs = model.predict(x_test)
    vol_size= K.int_shape(x_test)
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