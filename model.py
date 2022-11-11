from vae import AutoEncoder

class model():
    def __init__(self,
                 shape1d=[10000,1],
                 shape2d=[100,100,1],
                 ):
        self.vae1d = AutoEncoder(
            input_shape=shape1d,
            conv_filters=(32,64,64,64),
            conv_kernels=(3,3,3,3),
            conv_strides=(1,2,2,1),
            latent_dim=2,
        )
        self.vae2d = AutoEncoder(
            input_shape=shape2d,
            conv_filters=(32,64,64,64),
            conv_kernels=(3,3,3,3),
            conv_strides=(1,2,2,1),
            latent_dim=2,
        )
        
        