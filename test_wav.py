from vae import AutoEncoder, calculate_mse_loss
from constants import *
from datasets import *
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import museval
from evaluate import new_sdr

def test_loss(y, y_pred):
    error = y - y_pred
    reconstruction_loss = np.mean(np.square(error))
    return reconstruction_loss

def tensor_to_nparray(t):
    proto = tf.make_tensor_proto(t)
    nparray = tf.make_ndarray(proto)
    return nparray

if __name__ == '__main__':
    x_test, y_test = load_test_spectrums()
    #x_train, y_train = load_train_datasets()
    print(test_loss(y_test,np.zeros(y_test.shape)))
    print(np.mean(y_test))
    exit(0)
    model = AutoEncoder.load(MODEL_SAVEDIR_WAV)
    y = model.reconstruct("i:/dl/A Classic Education - NightOwl.stem.wav")
    print(y)
    print(y.shape)
    exit()
    y_pred = model.predict(x_test)
    loss = MeanSquaredError()(y_test, y_pred)
    print(loss)

    sdr = new_sdr(y_test, y_pred)
    print(sdr)
    sdr = new_sdr(y_test, y_test)
    print(sdr)