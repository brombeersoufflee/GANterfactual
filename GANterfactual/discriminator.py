from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.models import Model
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


def build_discriminator(img_shape, df):
    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            # https://stackoverflow.com/questions/68088889/how-to-add-instancenormalization-on-tensorflow-keras
            d = BatchNormalization(axis=[0,1])(d)
        return d

    img = Input(shape=img_shape)

    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, validity)

"""Me :With img_shape = (512,512, 1) and df = 64, what is the output shape of the model? Please give me the output shapes per d_layer in the CNN :)
GPT: Got it! If the image is grayscale, then the input shape is`(512, 512, 1)`.

Let's go through the layers again, but now with the grayscale input:

### Starting Parameters:

* `img_shape = (512, 512, 1)` (grayscale image with a single channel)
* `df = 64` (this defines the number of filters in the first convolutional layer)

### Layer-by-layer calculation:

#### Layer 1 (`d1`):

* **Input shape**: `(512, 512, 1)`
* Apply `Conv2D` with:

  * `filters = 64`
  * `kernel_size = (4, 4)`
  * `strides = 2`
  * `padding = 'same'`

  Using the formula for output dimensions:

  $$
  \text{Output width/height} = \left\lceil \frac{512}{2} \right\rceil = 256
  $$
* **Output shape**: `(256, 256, 64)`

#### Layer 2 (`d2`):

* **Input shape**: `(256, 256, 64)`
* Apply `Conv2D` with:

  * `filters = 128`
  * `kernel_size = (4, 4)`
  * `strides = 2`
  * `padding = 'same'`

  Using the formula for output dimensions:

  $$
  \text{Output width/height} = \left\lceil \frac{256}{2} \right\rceil = 128
  $$
* **Output shape**: `(128, 128, 128)`

#### Layer 3 (`d3`):

* **Input shape**: `(128, 128, 128)`
* Apply `Conv2D` with:

  * `filters = 256`
  * `kernel_size = (4, 4)`
  * `strides = 2`
  * `padding = 'same'`

  Using the formula for output dimensions:

  $$
  \text{Output width/height} = \left\lceil \frac{128}{2} \right\rceil = 64
  $$
* **Output shape**: `(64, 64, 256)`

#### Layer 4 (`d4`):

* **Input shape**: `(64, 64, 256)`
* Apply `Conv2D` with:

  * `filters = 512`
  * `kernel_size = (4, 4)`
  * `strides = 2`
  * `padding = 'same'`

  Using the formula for output dimensions:

  $$
  \text{Output width/height} = \left\lceil \frac{64}{2} \right\rceil = 32
  $$
* **Output shape**: `(32, 32, 512)`

#### Final Layer (`validity`):

* **Input shape**: `(32, 32, 512)`
* Apply `Conv2D` with:

  * `filters = 1` (to output a single validity score)
  * `kernel_size = (4, 4)`
  * `strides = 1`
  * `padding = 'same'`

  Using the formula for output dimensions:

  $$
  \text{Output width/height} = \left\lceil \frac{32}{1} \right\rceil = 32
  $$
* **Output shape**: `(32, 32, 1)`

### Summary of Output Shapes per `d_layer`:

1. **d1 (first Conv2D)**: `(256, 256, 64)`
2. **d2 (second Conv2D)**: `(128, 128, 128)`
3. **d3 (third Conv2D)**: `(64, 64, 256)`
4. **d4 (fourth Conv2D)**: `(32, 32, 512)`
5. **validity (final Conv2D)**: `(32, 32, 1)`

So with a grayscale image as input, the final output shape is still `(32, 32, 1)`.
"""