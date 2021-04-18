import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K

def find(model, x_test, cfg, storage):
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
        original = (x_test[i])
        recon = (decoded[i])

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
    plt.title("(Anomaly detection) Orginal vs. recunstructed")
    plt.show()