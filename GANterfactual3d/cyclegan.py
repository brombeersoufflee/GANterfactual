from __future__ import print_function, division

import datetime
import os

import keras
import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize

from keras.layers import Input, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


from classifier import load_classifier
from dataloader import DataLoader
from discriminator import build_discriminator
from generator import build_generator

class CycleGAN():

    def __init__(self):
        # Input shape
        # images modified in preprocessor.py to 512x512
        self.img_rows = 512
        self.img_cols = 512
        # greyscale images
        self.channels = 1
        #image shape defintion, for CNN
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Calculate output shape of D (PatchGAN)
        # TODO: what is PatchGAN?
        # PatchGAN: uses patches, so the discriminator will classify each patch as real or fake
        # Patch size is half of the image size, so 512/2^4 = 32 in this case
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1) # disc to be used on image in discriminator

        # Number of filters in the first layer of G and D
        # the generator and discriminator are both CNNs, so they have filters
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0  # Cycle-consistency loss lamda value (how much weight to give to the cycle-consistency loss)
        # Cycle-consistency loss: ensures that the translation from one domain to another and back results in the original image
        self.lambda_id = 0.1 * self.lambda_cycle  # Identity loss lambda value (how much weight to give to the identity loss)
        # Identity loss: ensures that the generator does not change the image if it is already in the target domain
        # TODO: Why  is the identityy value one thenth of the cycle-consistency loss?

        # Initialize the discriminators and generators
        self.d_N = None # Discriminator for NEGATIVE domain --> CNN classifier
        self.d_P = None # Discriminator for POSITIVE domain --> CNN classifier
        self.g_NP = None # Generator to translate NEGATIVE to POSITIVE domain -> CNN U Net
        # U-Net: a type of CNN that is used for image segmentation, but here used to translate images from one domain to another
        self.g_PN = None # Generator to translate POSITIVE to NEGATIVE domain -> CNN U Net
        self.combined = None # Combined model for training the generators -> GAN
        self.classifier = None # Classifier to classify the images in the two domains

    def construct(self, classifier_path=None, classifier_weight=None):
        # Build the discriminators
        # TODO: what is the difference between the two discriminators?
        # IDEA: The difference is the input images, one is for the NEGATIVE domain and the other for the POSITIVE domain
        # IDEA: There could be just one but the output is: Is this a real POSITIVE image or a fake POSITIVE image?
        # IDEA: The other one is: Is this a real NEGATIVE image or a fake NEGATIVE image?
        # The reasoning is that the discriminator is trained to classify images as real or fake, so it needs to know which domain the image is from
        # furthermore, both generators neeed their own adversaries, to make them better at generating images
        self.d_N = build_discriminator(self.img_shape, self.df)
        self.d_P = build_discriminator(self.img_shape, self.df) 

        # Build the generators
        self.g_NP = build_generator(self.img_shape, self.gf, self.channels)
        self.g_PN = build_generator(self.img_shape, self.gf, self.channels)

        # Combined model trains generators to fool discriminators
        # contains the losses for the generators and discriminators
        self.build_combined(classifier_path, classifier_weight)

    def load_existing(self, cyclegan_folder, classifier_path=None, classifier_weight=None):
        # The discriminators and generators from disk
        # the cycle itself is not loaded from disk, but built again
        # custom objects for loading the models, InstanceNormalization is a custom layer used in the generators and discriminators
        # It is used to normalize the activations of the previous layer at each batch, which helps to stabilize the training
        # To achieve this we use the keras
        custom_objects = {"InstanceNormalization": BatchNormalization(axis=[0,1])}

        # Load discriminators from disk
        self.d_N = keras.models.load_model(os.path.join(cyclegan_folder, 'discriminator_n.keras'),
                                           custom_objects=custom_objects)
        self.d_N._name = "d_N"
        self.d_P = keras.models.load_model(os.path.join(cyclegan_folder, 'discriminator_p.keras'),
                                           custom_objects=custom_objects)
        self.d_P._name = "d_P"

        # Load generators from disk
        self.g_NP = keras.models.load_model(os.path.join(cyclegan_folder, 'generator_np.keras'),
                                            custom_objects=custom_objects)
        self.g_NP._name = "g_NP"
        self.g_PN = keras.models.load_model(os.path.join(cyclegan_folder, 'generator_pn.keras'),
                                            custom_objects=custom_objects)
        self.g_PN._name = "g_PN"

        # builds combined model based on loaded discriminators and generators
        self.build_combined(classifier_path, classifier_weight)

    def save(self, cyclegan_folder):
        # save the generators and discriminators to disk, the cycle gan itself is not saved but built again (see above)
        os.makedirs(cyclegan_folder, exist_ok=True)

        # Save discriminators to disk
        self.d_N.save(os.path.join(cyclegan_folder, 'discriminator_n.keras'))
        self.d_P.save(os.path.join(cyclegan_folder, 'discriminator_p.keras'))

        # Save generators to disk
        self.g_NP.save(os.path.join(cyclegan_folder, 'generator_np.keras'))
        self.g_PN.save(os.path.join(cyclegan_folder, 'generator_pn.keras'))

    def build_combined(self, classifier_path=None, classifier_weight=None):
        # optimizer for both discriminators and generators, all use the same optimizer
        optimizer = Adam(0.0002, 0.5)

        # discriminators d_N / D_P means the discriminator for the NEGATIVE / POSITIVE domain
        # mean square error for the classifier, the classifier just returns one probability per image
        # TODO: Is 0 real or is 1 real?
        # TODO: assess accuracy because? why not balanced accuracy?
        self.d_N.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_P.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        # Input images from both domains
        # the input images are separated in negative and positive images
        img_N = Input(shape=self.img_shape)
        img_P = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_P = self.g_NP(img_N)
        fake_N = self.g_PN(img_P)
        # Translate images back to original domain
        reconstr_N = self.g_PN(fake_P)
        reconstr_P = self.g_NP(fake_N)
        # Identity mapping of images
        img_N_id = self.g_PN(img_N)
        img_P_id = self.g_NP(img_P)

        # For the combined model we will only train the generators
        # TODO: Why is this done?
        # GPT: The discriminators are already trained, so we only need to train the generators to fool the discriminators
        # TODO: Why are the generators not set to non-trainable? (GPT suggestts to also set the generators to non-trainable)
        self.d_N.trainable = False
        self.d_P.trainable = False

        # Discriminators determines validity of translated images
        valid_N = self.d_N(fake_N)
        valid_P = self.d_P(fake_P)

        if classifier_path is not None and os.path.isfile(classifier_path):
            self.classifier = load_classifier(classifier_path)
            self.classifier._name = "classifier"
            self.classifier.trainable = False

            class_N_loss = self.classifier(fake_N)
            class_P_loss = self.classifier(fake_P)

            # Combined model trains generators to fool discriminators
            self.combined = Model(inputs=[img_N, img_P],
                                  outputs=[valid_N, valid_P,
                                           class_N_loss, class_P_loss,
                                           reconstr_N, reconstr_P,
                                           img_N_id, img_P_id])

            self.combined.compile(loss=['mse', 'mse',
                                        'mse', 'mse',
                                        'mae', 'mae',
                                        'mae', 'mae'],
                                  loss_weights=[1, 1,
                                                classifier_weight, classifier_weight,
                                                self.lambda_cycle, self.lambda_cycle,
                                                self.lambda_id, self.lambda_id],
                                  optimizer=optimizer)

        else:
            # Combined model trains generators to fool discriminators
            self.combined = Model(inputs=[img_N, img_P],
                                  outputs=[valid_N, valid_P,
                                           reconstr_N, reconstr_P,
                                           img_N_id, img_P_id])

            self.combined.compile(loss=['mse', 'mse',
                                        'mae', 'mae',
                                        'mae', 'mae'],
                                  loss_weights=[1, 1,
                                                self.lambda_cycle, self.lambda_cycle,
                                                self.lambda_id, self.lambda_id],
                                  optimizer=optimizer)

    def train(self, dataset_name, epochs, batch_size=1, train_N="NEGATIVE", train_P="POSITIVE", print_interval=100,
              sample_interval=1000):

        # Configure data loader
        data_loader = DataLoader(dataset_name=dataset_name, img_res=(self.img_rows, self.img_cols))

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        class_N = np.stack([np.ones(batch_size), np.zeros(batch_size)]).T
        class_P = np.stack([np.zeros(batch_size), np.ones(batch_size)]).T

        for epoch in range(epochs): # epochs = 20
            for batch_i, (imgs_N, imgs_P) in enumerate(data_loader.load_batch(train_N, train_P, batch_size)): #load_batch yields a batch amount of (normal, pneumo)
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_P = self.g_NP.predict(imgs_N)
                fake_N = self.g_PN.predict(imgs_P)

                # Train the discriminators (original images = real / translated = Fake)
                dN_loss_real = self.d_N.train_on_batch(imgs_N, valid)
                dN_loss_fake = self.d_N.train_on_batch(fake_N, fake)
                dN_loss = 0.5 * np.add(dN_loss_real, dN_loss_fake)

                dP_loss_real = self.d_P.train_on_batch(imgs_P, valid)
                dP_loss_fake = self.d_P.train_on_batch(fake_P, fake)
                dP_loss = 0.5 * np.add(dP_loss_real, dP_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dN_loss, dP_loss)

                # ------------------
                #  Train Generators
                # ------------------

                if self.classifier is not None:
                    # Train the generators
                    g_loss = self.combined.train_on_batch([imgs_N, imgs_P],
                                                          [valid, valid,
                                                           class_N, class_P,
                                                           imgs_N, imgs_P,
                                                           imgs_N, imgs_P])
                else:
                    g_loss = self.combined.train_on_batch([imgs_N, imgs_P],
                                                          [valid, valid,
                                                           imgs_N, imgs_P,
                                                           imgs_N, imgs_P])

                elapsed_time = datetime.datetime.now() - start_time

                if self.classifier is not None:
                    progress_str = f"[Epoch: {epoch}/{epochs}] [Batch: {batch_i}] [D_loss: {d_loss[0]:.5f}, acc: {100 * d_loss[1]:.5f}] " \
                                   f"[G_loss: {g_loss[0]:.5f}, adv: {np.mean(g_loss[1:3]):.5f}, classifier_N: {g_loss[3]:.5f}, classifier_P: {g_loss[4]:.5f}, " \
                                   f"recon: {np.mean(g_loss[5:7]):.5f}, id: {np.mean(g_loss[7:9]):.5f}] " \
                                   f"time: {elapsed_time}"
                else:
                    progress_str = f"[Epoch: {epoch}/{epochs}] [Batch: {batch_i}] [D_loss: {d_loss[0]:.5f}, acc: {100 * d_loss[1]:.5f}] " \
                                   f"[G_loss: {g_loss[0]:.5f}, adv: {np.mean(g_loss[1:3]):.5f}, recon: {np.mean(g_loss[3:5]):.5f}, id: {np.mean(g_loss[5:7]):.5f}] " \
                                   f"time: {elapsed_time}"

                # Plot the progress
                if batch_i % print_interval == 0:
                    print(progress_str)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, imgs_N[0], imgs_P[0])

            # Comment this in if you want to save checkpoints:
            #self.save(os.path.join('..','models','GANterfactual','ep_' + str(epoch)))

    def sample_images(self, epoch, batch_i, testN, testP):
        os.makedirs('images', exist_ok=True)
        r, c = 2, 3

        img_N = testN[np.newaxis, :, :, :]
        img_P = testP[np.newaxis, :, :, :]

        # Translate images to the other domain
        fake_P = self.g_NP.predict(img_N)
        fake_N = self.g_PN.predict(img_P)
        # Translate back to original domain
        reconstr_N = self.g_PN.predict(fake_P)
        reconstr_P = self.g_NP.predict(fake_N)

        imgs = [img_N, fake_P, reconstr_N, img_P, fake_N, reconstr_P]
        classification = [['NEGATIVE', 'POSITIVE'][int(np.argmax(self.classifier.predict(x)))] for x in imgs]

        gen_imgs = np.concatenate(imgs)
        correct_classification = ['NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE']

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c, figsize=(15, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt][:, :, 0], cmap='gray')
                axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]} | {classification[cnt]})')
                axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]})')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d_%d.png" % (epoch, batch_i))
        plt.close()

    def predict(self, original_in_path, translated_out_path, reconstructed_out_path, force_original_aspect_ratio=False):
        assert (self.classifier is not None)
        data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

        original = data_loader.load_single(original_in_path)
        original = original.reshape(1, original.shape[0], original.shape[1], original.shape[2])

        pred_original = self.classifier.predict(original)
        if int(np.argmax(pred_original)) == 0:
            print("PREDICTION -- NEGATIVE")
            translated = self.g_NP.predict(original)
            reconstructed = self.g_PN.predict(translated)
        else:
            print("PREDICTION -- POSITIVE")
            translated = self.g_PN.predict(original)
            reconstructed = self.g_NP.predict(translated)

        pred_translated = self.classifier.predict(translated)
        pred_reconstructed = self.classifier.predict(reconstructed)

        if force_original_aspect_ratio:
            orig_no_res = keras.preprocessing.image.load_img(original_in_path)
            translated = resize(translated[0], (orig_no_res.height, orig_no_res.width))
            reconstructed = resize(reconstructed[0], (orig_no_res.height, orig_no_res.width))
        else:
            translated = translated[0]
            reconstructed = reconstructed[0]

        data_loader.save_single(translated, translated_out_path)
        data_loader.save_single(reconstructed, reconstructed_out_path)

        return [pred_original, pred_translated, pred_reconstructed]


if __name__ == '__main__':
    gan = CycleGAN()
    gan.construct(classifier_path=os.path.join('..', 'models', 'classifier', 'model.keras'), classifier_weight=1)
    gan.train(dataset_name=os.path.join("..","data"), epochs=20, batch_size=1, print_interval=10,
          sample_interval=100)
    gan.save(os.path.join('..', 'models', 'GANterfactual'))
