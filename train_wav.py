from datasets import *
from vae import VAE
import tensorflow as tf
import numpy as np

SHAPE=(352800,1)
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 1000

MODEL_SAVEDIR = "./wav"

def train(x_train, y_train, x_val, y_val, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS):
    try:
        model = VAE.load(MODEL_SAVEDIR)
        print("Loaded saved model")
    except:
        model = VAE(
            input_shape=SHAPE,
            conv_filters=(2,4,8,16,32,16,8),
            conv_kernels=(7,3,5,7,3,3,1),
            conv_strides=(5,2,3,5,3,2,1),
            latent_dim=512,
        )
        print("Created new model")
    model.summary()
    model.compile(learning_rate=learning_rate)
    model.train(x_train, y_train, x_val, y_val, batch_size=batch_size, num_epochs=epochs)
    return model

if __name__ == '__main__':
    mix, voc = load_train_datasets()
    print("train sets loaded")
    testmix, testvoc = load_test_datasets()
    model = train(mix, voc, testmix, testvoc)
    model.save("./wav")

