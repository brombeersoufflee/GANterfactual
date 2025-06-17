import keras
from keras import Input, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling
from keras.layers import BatchNormalization
import numpy as np
from keras.regularizers import l2
import os
import tensorflow as tf
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs")
np.random.seed(1000)
dimension = 512

def get_adapted_alexNet():

    input = Input(shape=(dimension, dimension, 1))

    # 1st Convolutional Layer
    x = Conv2D(filters=96,
               kernel_size=(11, 11),
               strides=(4, 4),
               padding='valid',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001))(input)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation before passing it to the next layer
    x = BatchNormalization()(x)

    # 2nd Convolutional Layer
    x = Conv2D(filters=256,
               kernel_size=(11, 11),
               strides=(1, 1),
               padding='valid',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # 3rd Convolutional Layer
    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid') (x)
    x  = Activation('relu')(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # 4th Convolutional Layer
    x = Conv2D(filters=384,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # 5th Convolutional Layer
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid') (x)
    x = Activation('relu')(x)
    # Pooling
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # Passing it to a dense layer
    x = Flatten()(x)
    # 1st Dense Layer
    x = Dense(4096,
              input_shape=(dimension * dimension * 1, ),
              kernel_regularizer=l2(0.001),
              bias_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    # Add Dropout to prevent overfitting
    x = Dropout(0.4)(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # 2nd Dense Layer
    x = Dense(4096, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    # Add Dropout
    x = Dropout(0.4)(x)
    # Batch Normalisation
    x = BatchNormalization()(x)

    # 3rd Dense Layer
    x = Dense(1000, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    # Add Dropout
    x = Dropout(0.4)(x)
    # Batch Normalisation
    x = BatchNormalization()(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)

    opt = keras.optimizers.SGD(0.0001, 0.9)
    model = Model(input, x)
    model.compile(loss='mse',
                  metrics=['accuracy'],
                  optimizer=opt)
    return model


def get_data():
    # dimension = 512
    image_size = dimension
    # set the batch size, aka how many images to process at once
    batch_size = 32
    # Load data for training
    # modifications: Use image_dataset_from_directory instead of deprecated ImageDataGenerator
    train_data = keras.utils.image_dataset_from_directory(
        "../data/train",
        labels='inferred',
        label_mode='categorical',  # one-hot encoded labels, class a is 0 and class b is 1, aligns with negative and positive in folder structure
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=True
    )

    validation_data = keras.utils.image_dataset_from_directory(
        "../data/validation",
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=True
    )

    # Normalize: (lambda x: x / 127.5 - 1) can be applied as part of a preprocessing layer
    normalization_layer = Rescaling(1./127.5, offset=-1)

    train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
    validation_data = validation_data.map(lambda x, y: (normalization_layer(x), y))

    return train_data, validation_data

# Example model (assuming you have this function defined elsewhere)
model = get_adapted_alexNet()
model.summary()

# modifications: name data validation data instead of test (there is another test data folder)
train, validation = get_data()

# Callbacks
# modifications: store model as .keras file
check_point = keras.callbacks.ModelCheckpoint("classifier.keras", save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = keras.callbacks.EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True)

if __name__ == "__main__":
    # modifications: use fit and store model as .keras file
    #.fit_generator() function first accepts a batch of the dataset, then performs backpropagation on it, and then updates the weights in our model
    # fit_generator was deprecated, it was replaced by fit, which can take a generator directly as an input
    hist = model.fit(
        train,
        validation_data=validation,
        epochs=1000,
        callbacks=[check_point, early_stopping, tensorboard_callback],
    )

    model.save(os.path.join('..','models','classifier','model.keras'), include_optimizer=False)
