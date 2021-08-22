import datetime
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import *
import numpy as np
from deepswarm import anomalies
from deepswarm.dataset import prepare_dataset
from vizualization import painter

# Function to save generated plots to file
def save_plot(plt_name, plot):
    save_path = '../tests/manual_model/'
    plot.savefig(save_path + plt_name)

# Command to setup GPU card
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# TensorBoard settings
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# Command to run TensoarBoard
# tensorboard --logdir ./logs/fit/

# Training settings
EPOCHS = 3
BATCH = 32

# Contaminate dataset with anomalies (e.g. dataset with 99% of 1 digits and 1% of 3 digits)
validLabel = [1]
anomalyLabel = [0]
contamination = 0.05
test_size = 0.2
random_state = 42

# Autoencoder model setup
input_img = Input(shape=(28, 28, 1))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
#encoded = Dense(64, activation='relu')(encoded)
volumeSize = K.int_shape(encoded)
flatten = Flatten()(encoded)
flatten = Dense(16)(flatten)
encoder_model = tf.keras.Model(inputs=input_img, outputs=flatten, name='encoder')

latentInputs = Input(shape=(16,))
decoded = Dense(np.prod(volumeSize[1:]))(latentInputs)
decoded = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(decoded)
decoded = Dense(128, activation='relu')(decoded)
# encoded = Dense(784, activation='relu')(encoded)
decoded = Dense(1, activation='sigmoid')(decoded)
decoder_model = tf.keras.Model(inputs=latentInputs, outputs=decoded, name='decoder')

autoencoder = keras.Model(input_img, decoder_model(encoder_model(input_img)),name="autoencoder")
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(autoencoder.summary())

# Dataset setup
(x_train, x_test, y_train, y_test) = prepare_dataset(validLabel,
                                    anomalyLabel,
                                    contamination,
                                    test_size,
                                    True,
                                    random_state)


# Training the model
history = autoencoder.fit(x_train, x_train,
                epochs=EPOCHS,
                batch_size=BATCH,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tensorboard_callback])

autoencoder.save(filepath="../tests/manual_model")
encoder_model.save(filepath="../tests/manual_model/encoder_model")
decoder_model.save(filepath="../tests/manual_model/decoder_model")


# Predicting results on test dataset
decoded_imgs = autoencoder.predict(x_test)
score, acc, *all_metrices = autoencoder.evaluate(x_test, x_test, batch_size=BATCH)

test= K.int_shape(x_test)
print()
print('Score: %1.4f' % score)
print('Evaluation Accuracy: %1.2f%%' % (acc*100))

# Display and save graphs
plot_model(autoencoder, to_file='../tests/manual_model/autoencoder_plot.png', show_shapes=True, show_layer_names=True)
plot_model(encoder_model, to_file='../tests/manual_model/encoder_plot.png', show_shapes=True, show_layer_names=True)
plot_model(decoder_model, to_file='../tests/manual_model/decoder_plot.png', show_shapes=True, show_layer_names=True)

plt_loss = painter.training_loss(history, EPOCHS)
save_plot( 'plt_loss.png', plt_loss)
plt_loss.show()

plt_acc = painter.training_acc(history, EPOCHS)
save_plot( 'plt_acc.png', plt_acc)
plt_acc.show()

plt_recunstructed_results = painter.reconstructed_results(autoencoder, x_test)
save_plot('plt_recunstructed_results.png', plt_recunstructed_results)
plt_recunstructed_results.show()

plt_MAE_loss = painter.MAE_loss(autoencoder, x_train, True)
save_plot( 'plt_MAE_loss.png', plt_MAE_loss)
plt_MAE_loss.show()

plt_encoded_image = painter.encoded_image(autoencoder, encoder_model, x_test)
save_plot('plt_encoded_image.png', plt_encoded_image)
plt_encoded_image.show()

# Find anomalies in data
plt_anomalies = anomalies.find(autoencoder, x_test, y_test, 0.99, True, validLabel, anomalyLabel)
save_plot('plt_anomalies.png', plt_anomalies)
plt_anomalies.show()

# Evaluate model
roc_curve = anomalies.calculate_roc_curve(autoencoder, x_test, y_test, True, validLabel, anomalyLabel)
roc_curve.show()