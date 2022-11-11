from datasets import *
from constants import *
from vae import AutoEncoder
import tensorflow as tf
import numpy as np

# train spectrum data
def train(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS):
    try:
        model = AutoEncoder.load(MODEL_SAVEDIR_SPEC)
        print("Loaded saved model")
    except:
        model = AutoEncoder(
            input_shape=SPEC_SHAPE,
            conv_filters=(2,4,8,16,32,16,8),
            conv_kernels=(3,3,3,3,3,3,1),
            conv_strides=(1,1,2,2,2,2,1),
            latent_dim=128,
            vae_beta=0.00001,
            scale=1,
            bias=1,
        )
        print("Created new model")
    model.summary()
    model.compile(learning_rate=learning_rate)

    mix, voc = load_train_spectrums()
    testmix, testvoc = load_test_spectrums()
    print(mix.shape, voc.shape, testmix.shape, testvoc.shape)
    return
    model.train(mix, voc, testmix, testvoc, batch_size=batch_size, num_epochs=epochs)
    return model

if __name__ == '__main__':
    model = train()
    model.save(MODEL_SAVEDIR_SPEC)

