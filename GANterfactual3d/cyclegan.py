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


from dataloader import DataLoader
from discriminator import build_discriminator
from generator import build_generator

class CycleGAN():

    def __init__(self):
        # Input shape
        self.input_shape = (64, 128, 64)
        # greyscale images
        self.channels = 1
        #image shape defintion, for CNN
        self.img_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.channels)
        # Calculate output shape of D (PatchGAN)
        # PatchGAN: uses patches, so the discriminator will classify each patch as real or fake
        # Now we have a 64x128x64 image, so we need to make cubes
        # Any power of 2 is possible, but which?
        # previous implementation used 32x32x1 patches, that is 1024 pixels
        # We use 16x16x16 patches, that is 4096 pixels
        # TODO: Fine tune parameter
        patch_sizes = np.array(self.input_shape)//16 # 16x16x16 patches, same as original implementation

        self.disc_patch = (patch_sizes[0], patch_sizes[1], patch_sizes[2], 1)

        # Number of filters in the first layer of G and D
        # the generator and discriminator are both CNNs, so they have filters
        # original values are 32 and 64, but we use less
        # TODO: FINE TUNE
        self.gf = 16
        self.df = 32

        # Loss weights
        self.lambda_cycle = 10.0  # Cycle-consistency loss lamda value (how much weight to give to the cycle-consistency loss)
        # Cycle-consistency loss: ensures that the translation from one domain to another and back results in the original image
        self.lambda_id = 0.1 * self.lambda_cycle  # Identity loss lambda value (how much weight to give to the identity loss)
        # Identity loss: ensures that the generator does not change the image if it is already in the target domain
        # TODO: Why  is the identityy value one thenth of the cycle-consistency loss?

        # Initialize the discriminators and generators
        self.d_N = None # Discriminator for negative domain --> CNN classifier
        self.d_P = None # Discriminator for positive domain --> CNN classifier
        self.g_NP = None # Generator to translate negative to positive domain -> CNN U Net
        # U-Net: a type of CNN that is used for image segmentation, but here used to translate images from one domain to another
        self.g_PN = None # Generator to translate positive to negative domain -> CNN U Net
        self.combined = None # Combined model for training the generators -> GAN
        self.classifier = None # Classifier to classify the images in the two domains

    def construct(self, classifier_path=None, classifier_weight=None):
        # Build the discriminators
        # TODO: what is the difference between the two discriminators?
        # IDEA: The difference is the input images, one is for the negative domain and the other for the positive domain
        # IDEA: There could be just one but the output is: Is this a real positive image or a fake positive image?
        # IDEA: The other one is: Is this a real negative image or a fake negative image?
        # The reasoning is that the discriminator is trained to classify images as real or fake, so it needs to know which domain the image is from
        # furthermore, both generators neeed their own adversaries, to make them better at generating images
    
        # give name to avoid conflicts
        self.d_N = build_discriminator(self.img_shape, self.df, name = 'discriminator_negative')
        self.d_P = build_discriminator(self.img_shape, self.df, name = 'discriminator_positive') 

        # Build the generators
        self.g_NP = build_generator(self.img_shape, self.gf, self.channels, name='generator_neg_to_pos')
        self.g_PN = build_generator(self.img_shape, self.gf, self.channels, name='generator_pos_to_neg')

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
        self.d_N = keras.saving.load_model(os.path.join(cyclegan_folder, 'discriminator_n.keras'),
                                           custom_objects=custom_objects)
        self.d_N._name = "d_N"
        self.d_P = keras.saving.load_model(os.path.join(cyclegan_folder, 'discriminator_p.keras'),
                                           custom_objects=custom_objects)
        self.d_P._name = "d_P"

        # Load generators from disk
        self.g_NP = keras.saving.load_model(os.path.join(cyclegan_folder, 'generator_np.keras'),
                                            custom_objects=custom_objects)
        self.g_NP._name = "g_NP"
        self.g_PN = keras.saving.load_model(os.path.join(cyclegan_folder, 'generator_pn.keras'),
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

        # discriminators d_N / D_P means the discriminator for the negative / positive domain
        # mean square error for the classifier, the classifier just returns one probability per image
        # TODO: Is 0 real or is 1 real?
        # TODO: assess accuracy because? why not balanced accuracy?
        # TODO: Why do they use mse and not binary crossentropy?
        # GPT: MSE is a good loss function for regression tasks. Binary crossentropy could also be used
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
            # The following compile = False is essential to circumvent an error, it is also just used for inference so it is not needed compiled
            self.classifier = keras.saving.load_model(classifier_path, compile=False)
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

    def train(self, data_dir, epochs, batch_size=1, train_N="negative", train_P="positive", print_interval=100,
              sample_interval=1000):

        # Configure data loader
        data_loader = DataLoader(data_dir=data_dir, img_res=self.input_shape)

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
        # Add a new axis to the image to make it 5D, as the model expects a batch dimension
        # np.newaxis adds a new axis to the array, so the shape becomes (1, 64, 128, 64, 1)
        img_N = testN[np.newaxis, :, :, :, :]
        img_P = testP[np.newaxis, :, :, :, :]

        # Translate images to the other domain
        fake_P = self.g_NP.predict(img_N)
        fake_N = self.g_PN.predict(img_P)
        # Translate back to original domain
        reconstr_N = self.g_PN.predict(fake_P)
        reconstr_P = self.g_NP.predict(fake_N)

        imgs = [img_N, fake_P, reconstr_N, img_P, fake_N, reconstr_P]
        classification = [['negative', 'positive'][int(np.argmax(self.classifier.predict(x)))] for x in imgs]

        gen_imgs = np.concatenate(imgs)
        correct_classification = ['negative', 'positive', 'negative', 'positive', 'negative', 'positive']

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # this is for the created image sample
        r, c = 2, 3
        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c, figsize=(8, 12))
        cnt = 0
        for i in range(r):
            for j in range(c):
                # get middle slice for visualisation
                mid_slice = gen_imgs[cnt].shape[2] // 2
                axs[i, j].imshow(np.rot90(gen_imgs[cnt][:, :, mid_slice, 0], k=-1), cmap='gray')
                axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]} | {classification[cnt]})')
                axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]})')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d_%d.png" % (epoch, batch_i))
        plt.close()
    
    # not used?
    def predict(self, original_in_path, translated_out_path, reconstructed_out_path, force_original_aspect_ratio=False):
        assert (self.classifier is not None)
        data_loader = DataLoader(img_res=self.input_shape)

        original = data_loader.load_single(original_in_path)
        # original = original.reshape(1, original.shape[0], original.shape[1], original.shape[2])

        pred_original = self.classifier.predict(original)
        if int(np.argmax(pred_original)) == 0:
            print("PREDICTION -- negative")
            translated = self.g_NP.predict(original)
            reconstructed = self.g_PN.predict(translated)
        else:
            print("PREDICTION -- positive")
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
    print(os.getcwd())
    gan = CycleGAN()
    gan.construct(classifier_path=os.path.join('GANterfactual3d', 'classifier.keras'), classifier_weight=1)
    gan.train(data_dir=os.path.join("data3d"), epochs=20, batch_size=1, print_interval=10,
          sample_interval=100)
    gan.save(os.path.join('GANterfactual3d'))
