from tensorflow.keras.datasets import mnist

from vae import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100

def load_mnist():
    def normalize(x):
        x = x.astype("float32") / 255
        x = x.reshape(x.shape + (1,))
        return x
        
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    return x_train, y_train, x_test, y_test

def train(x_train, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS):
    model = VAE(
        input_shape=(28,28,1),
        conv_filters=(32,64,64,64),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_dim=2,
    )
    model.summary()
    model.compile(learning_rate=learning_rate)
    model.train(x_train, batch_size=batch_size, num_epochs=epochs)
    return model

if __name__ == '__main__':
    x_train, _,_,_ = load_mnist()

    model = train(x_train[:10000])
    model.save("model2d")
    model2 = VAE.load("model2d")
    model2.summary()
