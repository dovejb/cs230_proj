from datasets import load_datasets
from vae import VAE
import tensorflow as tf
import numpy as np

SHAPE=(352800,1)
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100

SAVEDIR = "./wav"

def train(x_train, y_train, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS):
    try:
        model = VAE.load(SAVEDIR)
    except:
        model = VAE(
            input_shape=SHAPE,
            conv_filters=(2,4,8,16,32,16,8),
            conv_kernels=(7,3,5,7,3,3,1),
            conv_strides=(5,2,3,5,3,2,1),
            latent_dim=512,
        )
    model.summary()
    model.compile(learning_rate=learning_rate)
    model.train(x_train, y_train, batch_size=batch_size, num_epochs=epochs)
    return model

if __name__ == '__main__':
    tf.random.set_seed(2021050300)
    mix, voc = load_datasets("i:/dl/train")
    mix = mix[..., np.newaxis]
    voc = voc[..., np.newaxis]
    model = train(mix, voc)
    model.save("./wav")

