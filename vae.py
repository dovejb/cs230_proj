import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os
import pickle
import museval
from constants import *
import soundfile as sf
import librosa
from constants import *
from preprocess import split_wav
from evaluate import new_sdr


# A metric to calculate MSE with all zero matrix
# because my y_true is too close to zero, and my model is trained to make
# y_pred close to zero, too
# but this seems already solved by bias
def mse_with_zero(y, y_pred):
    return K.mean(K.square(y_pred))

# These two loss functions are to calculate MSE and KL loss for VAE
# but my VAE model has gradient exploding
# so I use AutoEncoder now and this loss is not used.
def calculate_mse_loss(y, y_pred):
    error = y - y_pred
    reconstruction_loss = K.mean(K.square(error))
    return reconstruction_loss

def calculate_kl_loss(model):
    def _calculate_kl_loss(*args):
        kl_loss = -0.5 * K.sum(1 + model.log_variance - K.square(model.mu) -
                               K.exp(model.log_variance), axis=1)
        return kl_loss
    return _calculate_kl_loss


# AutoEncoder
# The conv & conv transpose layer are defined by parameters
# This is just for convenience
# And the dimension is decided by parameter, too.
# If input_shape is 2 dim, it uses Conv1D
class AutoEncoder():
    def __init__(self,
                 input_shape = [28, 28, 1], #[H,W,C]  if length is 2, use Conv1D
                 conv_filters = [2, 4, 8],
                 conv_kernels = [3, 5, 3],
                 conv_strides = [1, 2, 2],
                 latent_dim = 2,
                 vae_beta = 0.001,
                 scale = 1,     # This scales the input X&Y at the same time to solve the Y-close-to-zero issue, but the bias seems more useful
                 bias = 0,      # Same as above
                 ):

        self.input_shape = input_shape   
        self.conv_filters = conv_filters 
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim
        self.vae_beta = vae_beta
        self.scale = scale
        self.bias = bias

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
            self.vae_beta,
            self.scale,
            self.bias,
        ]
        with open(os.path.join(folder, "parameters.pkl"), "wb") as f:
            pickle.dump(parameters, f)
        self.model.save_weights(os.path.join(folder, "weights.h5"))
        
    @classmethod
    def load(cls, folder="."):
        with open(os.path.join(folder, "parameters.pkl"), "rb") as f:
            parameters = pickle.load(f)
        inst = AutoEncoder(*parameters)
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
        #self.log_variance = tfl.Dense(self.latent_dim, name="log_variance")(x)

        #def sample_point_from_normal_distribution(args):
        #    mu, log_variance = args
        #    epsilon = K.random_normal(shape=K.shape(self.mu), mean=0., stddev=1.)
        #    sample_points = mu + K.exp(log_variance/2) * epsilon
        #    return sample_points

        #output = tfl.Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])
        output = self.mu #back to Autoencoder and try

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
        self.model = Model(input, output, name="model")

    def loss(self, y, y_pred):
        mse_loss = calculate_mse_loss(y, y_pred)
        kl_loss = calculate_kl_loss(self)()
        return mse_loss + self.vae_beta * kl_loss

    def compile(self, learning_rate=0.002):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=MeanSquaredError(),
                            metrics=[mse_with_zero])
                           #loss=self.loss,
                           #metrics=[_calculate_reconstruction_loss, calculate_kl_loss(self)])

    def train(self, x_train, y_train, x_val, y_val, batch_size, num_epochs):
        y_train = y_train * self.scale + self.bias
        y_val = y_val * self.scale + self.bias
        x_train = x_train * self.scale + self.bias
        x_val = x_val * self.scale + self.bias
        self.model.fit(x_train, 
                       y_train, 
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True,
                       validation_data=(x_val, y_val))

    def forward(self, x):
        x = self.encoder(x)
        return x
    
    def predict(self, x):
        return (self.model.predict(x*self.scale+self.bias) - self.bias) / self.scale

    def eval(self, x, y_true):
        y = self.predict(x)
        return new_sdr(y_true, y)

    # load a wav file and reconstruct it using the model
    def reconstruct(self, path):
        seglen = self.input_shape[0]
        wav, orig_sr = sf.read(path)
        wavr = librosa.resample(wav, orig_sr=orig_sr, target_sr=SR)
        x = split_wav(wavr, seglen)
        y = self.predict(x)
        ry = np.reshape(y, (y.shape[0]*y.shape[1],y.shape[2]))[:wavr.size]
        r = librosa.resample(ry, orig_sr=SR, target_sr=orig_sr)
        assert wav.shape == r.shape
        sf.write("out.wav", r, orig_sr)
        print(np.mean(wav), np.mean(r))
        return r


if __name__ == "__main__":
    model = AutoEncoder(
        input_shape=(28,28,1),
        conv_filters=(32,64,64,64),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_dim=2,
    )
    model.summary()
