from datasets import *
from constants import *
from vae import VAE
import tensorflow as tf
import numpy as np

def train(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS):
    try:
        model = VAE.load(MODEL_SAVEDIR)
        print("Loaded saved model")
    except:
        model = VAE(
            input_shape=SHAPE,
            conv_filters=(2,4,8,8,16,16),
            conv_kernels=(3,3,3,3,3,3),
            conv_strides=(1,2,3,5,2,1),
            latent_dim=128,
            vae_beta=0.00001,
            scale=0.001
        )
        print("Created new model")
    model.summary()
    model.compile(learning_rate=learning_rate)

    mix, voc = load_train_datasets()
    testmix, testvoc = load_test_datasets()

    model.train(mix, voc, testmix, testvoc, batch_size=batch_size, num_epochs=epochs)
    return model

if __name__ == '__main__':
    model = train()
    model.save("./wav")

