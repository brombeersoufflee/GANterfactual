from keras.layers import Dropout
from keras.layers import Input, Concatenate
from keras.layers import LeakyReLU
from keras.layers import UpSampling3D, Conv3D
from keras.models import Model
from keras.layers import BatchNormalization


def build_generator(img_shape, gf, channels, name = "generator"):
    """U-Net Generator"""

    def conv3d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(negative_slope=0.2)(d)
        d = BatchNormalization()(d)
        return d

    def deconv3d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling3D(size=2)(layer_input)
        u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv3d(d0, gf)
    d2 = conv3d(d1, gf * 2)
    d3 = conv3d(d2, gf * 4)
    d4 = conv3d(d3, gf * 8)

    # Upsampling
    u1 = deconv3d(d4, d3, gf * 4)
    u2 = deconv3d(u1, d2, gf * 2)
    u3 = deconv3d(u2, d1, gf)

    u4 = UpSampling3D(size=2)(u3)
    output_img = Conv3D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, output_img, name=name)