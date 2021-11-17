import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, \
    BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model


class GAN():
    def __init__(self
                 , input_dim
                 , discriminator_learning_rate
                 , generator_learning_rate
                 , z_dim
                 ):

        self.name = 'gan'

        self.input_dim = input_dim
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate
        self.z_dim = z_dim

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.d_losses = []
        self.g_losses = []

        self.epoch = 0

        self._build_discriminator()
        self._build_generator()

        self._build_adversarial()

    def _build_discriminator(self):
        dropout_rate = 0.4
        discriminator_input = Input((28, 28, 1), name="discriminator_input")

        x = discriminator_input

        x = Conv2D(64, 5, strides=2, activation='relu', padding="same", kernel_initializer=self.weight_init, name="d_conv_0")(x)
        x = Activation('relu', name="d_act_0")(x)
        x = Dropout(dropout_rate, name="d_drop_0")(x)

        x = Conv2D(64, 5, strides=2, activation='relu', padding="same", kernel_initializer=self.weight_init, name="d_conv_1")(x)
        x = Activation('relu', name="d_act_1")(x)
        x = Dropout(dropout_rate, name="d_drop_1")(x)

        x = Conv2D(128, 5, strides=2, activation='relu', padding="same", kernel_initializer=self.weight_init, name="d_conv_2")(x)
        x = Activation('relu', name="d_act_2")(x)
        x = Dropout(dropout_rate, name="d_drop_2")(x)

        x = Conv2D(128, 5, strides=1, activation='relu', padding="same", kernel_initializer=self.weight_init, name="d_conv_3")(x)
        x = Activation('relu', name="d_act_3")(x)
        x = Dropout(dropout_rate, name="d_drop_3")(x)

        x = Flatten()(x)

        discriminator_output = Dense(1, activation="sigmoid", kernel_initializer=self.weight_init, name="d_dense_0")(x)

        self.discriminator = Model(discriminator_input, discriminator_output, name="discriminator")
        self.discriminator.summary()

    def _build_generator(self):

        generator_input = Input(shape=(self.z_dim,), name='generator_input')

        x = generator_input
        x = Dense(np.prod((7, 7, 64)), kernel_initializer=self.weight_init, name="g_dense_0")(x)

        x = BatchNormalization(momentum=0.9)(x)

        x = Activation('relu')(x)

        x = Reshape((7, 7, 64))(x)

        # no dropout

        # 0th pack
        # upsampling = 2
        x = UpSampling2D()(x)

        x = Conv2D(filters=128, kernel_size=5, padding='same', strides=1, kernel_initializer=self.weight_init, name="g_conv_0")(x)

        x = BatchNormalization(momentum=0.9)(x)

        x = Activation('relu')(x)

        # 1st pack
        # upsampling = 2
        x = UpSampling2D()(x)

        x = Conv2D(filters=64, kernel_size=5, padding='same', strides=1, kernel_initializer=self.weight_init, name="g_conv_1")(x)

        x = BatchNormalization(momentum=0.9)(x)

        x = Activation('relu')(x)

        # 2nd pack
        # upsampling = 1
        x = Conv2DTranspose(filters=64, kernel_size=5, padding='same', strides=1, kernel_initializer=self.weight_init, name="g_conv_2")(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        # 3nd pack
        # upsampling = 1
        x = Conv2DTranspose(filters=1, kernel_size=5, padding='same', strides=1, kernel_initializer=self.weight_init, name="g_conv_3")(x)
        x = Activation('tanh')(x)

        generator_output = x

        self.generator = Model(generator_input, generator_output, name="generator")
        self.generator.summary()

    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def _build_adversarial(self):

        ### COMPILE DISCRIMINATOR

        self.discriminator.compile(
            optimizer=RMSprop(self.discriminator_learning_rate)
            , loss='binary_crossentropy'
            , metrics=['accuracy']
        )

        ### COMPILE THE FULL GAN

        self.set_trainable(self.discriminator, False)

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)

        self.model.compile(optimizer=RMSprop(self.generator_learning_rate), loss='binary_crossentropy',
                           metrics=['accuracy']
                           , experimental_run_tf_function=False
                           )

        self.set_trainable(self.discriminator, True)

    def train_discriminator(self, x_train, batch_size, using_generator):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real, d_acc_real = self.discriminator.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self, x_train, batch_size, epochs, run_folder
              , print_every_n_batches=50
              , using_generator=False):

        for epoch in range(self.epoch, self.epoch + epochs):

            d = self.train_discriminator(x_train, batch_size, using_generator)
            g = self.train_generator(batch_size)

            print("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (
            epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))

            self.d_losses.append(d)
            self.g_losses.append(g)

            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
                self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                self.save_model(run_folder)

            self.epoch += 1

    def sample_images(self, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.discriminator, to_file=os.path.join(run_folder, 'viz/discriminator.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.generator, to_file=os.path.join(run_folder, 'viz/generator.png'), show_shapes=True,
                   show_layer_names=True)

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.discriminator.save(os.path.join(run_folder, 'discriminator.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
