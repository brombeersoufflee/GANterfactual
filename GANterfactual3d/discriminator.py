from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Conv3D
from keras.layers import BatchNormalization
from keras.models import Model


def build_discriminator(img_shape, df, name = "discriminator"):
    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(negative_slope=0.2)(d)
        if normalization:
            # https://stackoverflow.com/questions/68088889/how-to-add-instancenormalization-on-tensorflow-keras
            # TODO: dive into the math and make sure this is correct
            # TODO: There are more BatchNormalizations in the code base so if u replace this one then replace all of them
            d = BatchNormalization()(d)
        return d

    img = Input(shape=img_shape)

    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    # The output shape needs to match the "valid" parameter shape in the cyclegan
    validity = Conv3D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, validity, name=name)
