from __future__ import print_function, division
import os
import keras
from keras.layers import Rescaling


class DataLoader():
    def __init__(self, dataset_name=None, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    # needs to be modified to be "negative" and "positive" for the two classes, not capitalized
    def load_batch(self, train_N="negative", train_P="positive", batch_size=16, is_testing=False):
        
        subdir = "validation" if is_testing else "train"

        # Normalize: (lambda x: x / 127.5 - 1) can be applied as part of a preprocessing layer
        normalization_layer = Rescaling(1./127.5, offset=-1)
        
        # Load negative (class 0) images only
        data_negative = keras.utils.image_dataset_from_directory(
            f"../data/{subdir}/{train_N}",
            labels=None,
            color_mode='grayscale',
            batch_size=batch_size,
            image_size=self.img_res,
            shuffle=True
        ).map(lambda x: normalization_layer(x))

        data_positive = keras.utils.image_dataset_from_directory(
            f"../data/{subdir}/{train_P}",
            labels=None,
            color_mode='grayscale',
            batch_size=batch_size,
            image_size=self.img_res,
            shuffle=True
        ).map(lambda x: normalization_layer(x))

        # endless loop so we can use the maximum
        n_batches = max(len(data_negative), len(data_positive))

        for normal, pneumo, _ in zip(data_negative, data_positive, range(n_batches)):
            yield normal, pneumo

    def load_single(self, path):
        img = keras.preprocessing.image.load_img(path, color_mode="grayscale", target_size=self.img_res)
        x = keras.preprocessing.image.img_to_array(img) / 127.5 - 1
        return x

    def save_single(self, x, path):
        # Rescale images 0 - 1
        x = 0.5 * x + 0.5
        keras.preprocessing.image.save_img(path, x)