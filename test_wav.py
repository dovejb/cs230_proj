from vae import VAE, _calculate_reconstruction_loss
from constants import *
from datasets import load_test_datasets
import tensorflow as tf
import numpy as np

def test_loss(y, y_pred):
    error = y - y_pred
    reconstruction_loss = np.mean(np.square(error))
    return reconstruction_loss

def tensor_to_nparray(t):
    proto = tf.make_tensor_proto(t)
    nparray = tf.make_ndarray(proto)
    return nparray

if __name__ == '__main__':
    model = VAE.load(MODEL_SAVEDIR)
    model.summary()
    x_test, y_test = load_test_datasets()
    y_pred = model.predict(x_test)
    print(test_loss(y_test, y_pred))
    print(test_loss(y_test, y_test))
    print(test_loss(x_test, y_test))
    print(test_loss(np.zeros((y_test.shape)), y_test))