import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from deepswarm.log import Log
from . import data_config

# Good tutorial https://www.pyimagesearch.com/2020/03/02/anomaly-detection-with-keras-tensorflow-and-deep-learning/
def find(model, x_test, y_test, quantile=0.99,
         manual_code=False,
         valid_label=data_config['valid_label'],
         anomaly_label=data_config['anomaly_label']):


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
    thresh = np.quantile(errors, quantile)
    idxs = np.where(np.array(errors) >= thresh)[0]

    response_msg_1 = f"[INFO] mse threshold: {thresh}"
    response_msg_2 = f"[GOAL] Actual number of all anomalies in dataset: {sum(x in anomaly_label for x in y_test)}"
    instaces_in_quantile = np.array(y_test)[idxs.astype(int)]
    TP_anomalies = sum(x == anomaly_label for x in instaces_in_quantile)
    response_msg_3 = f"[RESULT] Number of true positives anomalies found: {sum(TP_anomalies)} instances inside of quantile:{quantile}"

    if manual_code:
        print(response_msg_1)
        print(response_msg_2)
        print(response_msg_3)
    else:
        Log.info(response_msg_1)
        Log.info(response_msg_2)
        Log.info(response_msg_3)

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
    final_output = outputs.reshape(vol_size[0], vol_size[1])
    plt.figure(figsize = (5,len(idxs)+5))
    plt.imshow(final_output)
    plt.title(f"(Anomaly detection)\n"
              f"Model found: {len(idxs)} instances inside of quantile:{quantile}\n"
              f"Number of true positives anomalies: {TP_anomalies}\n"
              f"Valid label/s: {valid_label}\n"
              f"Anomaly label/s: {anomaly_label}")
    return plt