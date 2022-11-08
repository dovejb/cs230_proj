import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os
import pickle


tf.compat.v1.disable_eager_execution()

def _calculate_reconstruction_loss(y, y_pred):
    error = y - y_pred
    reconstruction_loss = K.mean(K.square(error))
    return reconstruction_loss

def calculate_kl_loss(model):
    def _calculate_kl_loss(*args):
        kl_loss = -0.5 * K.sum(1 + model.log_variance - K.square(model.mu) -
                               K.exp(model.log_variance), axis=1)
        return kl_loss
    return _calculate_kl_loss

class VAE():
    def __init__(self,
                 input_shape = [28, 28, 1], #[H,W,C]  if length is 2, use Conv1D
                 conv_filters = [2, 4, 8],
                 conv_kernels = [3, 5, 3],
                 conv_strides = [1, 2, 2],
                 latent_dim = 2,
                 vae_alpha = 1000,
                 ):

        self.input_shape = input_shape   
        self.conv_filters = conv_filters 
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim
        self.vae_alpha = vae_alpha

        self._conv_layer = tfl.Conv2D
        self._convtranspose_layer = tfl.Conv2DTranspose
        if len(input_shape) == 2:
            self._conv_layer = tfl.Conv1D
            self._convtranspose_layer = tfl.Conv1DTranspose

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def save(self, folder="."):
        if not os.path.exists(folder):
            os.makedirs(folder)
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_dim,
            self.vae_alpha,
        ]
        with open(os.path.join(folder, "parameters.pkl"), "wb") as f:
            pickle.dump(parameters, f)
        self.model.save_weights(os.path.join(folder, "weights.h5"))
        
    @classmethod
    def load(cls, folder="."):
        with open(os.path.join(folder, "parameters.pkl"), "rb") as f:
            parameters = pickle.load(f)
        inst = VAE(*parameters)
        inst.model.load_weights(os.path.join(folder, "weights.h5"))
        return inst

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_model()

    def _build_encoder(self):
        input = tfl.Input(shape=self.input_shape, name='encoder_input')
        self._model_input = input
        x = input

        # Conv layers
        for i in range(self._num_conv_layers):
            layer_no = i+1
            x = self._conv_layer(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernels[i],
                strides=self.conv_strides[i],
                padding='same',
                name=f"encoder_conv_{layer_no}",
            )(x)
            x = tfl.ReLU(name=f"encoder_relu_{layer_no}")(x)
            x = tfl.BatchNormalization(name=f"encoder_batchnorm_{layer_no}")(x)

        self._shape_before_bottleneck = K.int_shape(x)[1:]

        # Flatten
        x = tfl.Flatten()(x)
        self.mu = tfl.Dense(self.latent_dim, name="mu")(x)
        self.log_variance = tfl.Dense(self.latent_dim, name="log_variance")(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0., stddev=1.)
            sample_points = mu + K.exp(log_variance/2) * epsilon
            return sample_points

        output = tfl.Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])

        self._model_input = input
        self.encoder = Model(input, output, name="encoder")

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build_decoder(self):
        input = tfl.Input(self.latent_dim, name='decoder_input')
        x = input
        
        # extent latent
        x = tfl.Dense(np.prod(self._shape_before_bottleneck), name='decoder_dense')(x)

        # reshape
        x = tfl.Reshape(self._shape_before_bottleneck)(x)

        # Conv transpose
        for i in reversed(range(1, self._num_conv_layers)):
            layer_no = self._num_conv_layers - i
            x = self._convtranspose_layer(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernels[i],
                strides=self.conv_strides[i],
                padding='same',
                name=f"decoder_convtrans_layer_{layer_no}",
            )(x)
            x = tfl.ReLU(name=f"decoder_relu_{layer_no}")(x)
            x = tfl.BatchNormalization(name=f"decoder_batchnorm_{layer_no}")(x)

        # Conv transpose output
        x = self._convtranspose_layer(
            filters=self.input_shape[-1],
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding='same',
            name=f"decoder_convtrans_layerr_{self._num_conv_layers}",
        )(x)
        output = tfl.Activation("sigmoid", name="sigmoid_layer")(x)

        self.decoder = Model(input, output, name='decoder')

    def _build_model(self):
        input = self._model_input
        output = self.decoder(self.encoder(input))
        self.model = Model(input, output, name="VAE2D")

    def loss(self, y, y_pred):
        reconstruction_loss = _calculate_reconstruction_loss(y, y_pred)
        kl_loss = calculate_kl_loss(self)()
        return self.vae_alpha * reconstruction_loss + kl_loss

    def compile(self, learning_rate=0.002):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self.loss,
                           metrics=[_calculate_reconstruction_loss, calculate_kl_loss(self)])

    def train(self, x_train, y_train, batch_size, num_epochs):
        self.model.fit(x_train, 
                       y_train, 
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

    def forward(self, x):
        x = self.encoder(x)
        return x

if __name__ == "__main__":
    model = VAE(
        input_shape=(28,28,1),
        conv_filters=(32,64,64,64),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_dim=2,
    )
    model.summary()
