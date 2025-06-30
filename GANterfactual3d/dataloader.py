from __future__ import print_function, division
import os
from keras.layers import Rescaling
import numpy as np
import tensorflow as tf


class DataLoader():
    def __init__(self, data_dir="data3d/", img_res=(64, 128, 64)):
        self.data_dir = data_dir
        self.img_res = img_res

    def load_npy_volume(self, file_path):
        volume = np.load(file_path)
        volume = tf.convert_to_tensor(volume, dtype=tf.int8)

        # Reshape if needed (make sure it has shape D x H x W x 1)
        if len(volume.shape) == 3:
            volume = tf.expand_dims(volume, axis=-1)

        # Resize to match desired resolution (optional if files are already correct size)
        # volume = tf.image.resize(volume, self.img_res[1:3])  # resize height x width
        # volume = tf.image.resize(volume, [self.img_res[0]], method="nearest")  # resize depth

        return volume

    def get_dataset_from_directory(self, folder_path, batch_size, normalization_layer):
        # List all .npy files
        file_names = os.listdir(folder_path)
        np.random.shuffle(file_names)

        def gen():
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)
                vol = self.load_npy_volume(file_path)
                vol = normalization_layer(vol)
                yield vol

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=tf.TensorSpec(shape=(*self.img_res, 1), dtype=tf.float32)
        )

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def load_batch(self, train_N="negative", train_P="positive", batch_size=16, is_testing=False):
        subdir = "validation" if is_testing else "train"

        normalization_layer = Rescaling(1./127.5, offset=-1)

        data_negative = self.get_dataset_from_directory(
            os.path.join(self.data_dir, subdir, train_N),
            batch_size,
            normalization_layer
        )

        data_positive = self.get_dataset_from_directory(
            os.path.join(self.data_dir, subdir, train_P),
            batch_size,
            normalization_layer
        )

        # Endless loop to match the original behavior
        n_batches = max(len(list(data_negative)), len(list(data_positive)))

        for normal, pneumo, _ in zip(data_negative, data_positive, range(n_batches)):
            yield normal, pneumo
    
    def load_single(self, path):
        vol = np.load(path)

        if len(vol.shape) == 3:
            vol = np.expand_dims(vol, axis=-1)  # Add channel dim

        # vol = tf.image.resize(vol, self.img_res[1:3])  # Resize HxW
        # vol = tf.image.resize(vol, [self.img_res[0]], method="nearest")  # Resize D
        vol = tf.convert_to_tensor(vol, dtype=tf.int8)
        return tf.reshape(vol, (1, *self.img_res, 1))
    
    def save_single(self, x, path):
        # Rescale images 0 - 1
        x = 0.5 * x + 0.5
        np.save(path, x)