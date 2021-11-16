import os

import matplotlib.pyplot as plt

from GAN import GAN
from loaders import load_safari

# run params
SECTION = 'gan'
RUN_ID = '0001'
DATA_NAME = 'camel'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode = 'build'  # 'load' #

x_train = load_safari()

plt.imshow(x_train[200, :, :, 0], cmap='gray')
plt.show()

gan = GAN(input_dim=(28, 28, 1)
          , discriminator_learning_rate=0.0008
          , generator_initial_dense_layer_size=(7, 7, 64)
          , generator_upsample=[2, 2, 1, 1]
          , generator_conv_filters=[128, 64, 64, 1]
          , generator_conv_kernel_size=[5, 5, 5, 5]
          , generator_conv_strides=[1, 1, 1, 1]
          , generator_batch_norm_momentum=0.9
          , generator_activation='relu'
          , generator_dropout_rate=None
          , generator_learning_rate=0.0004
          , optimiser='rmsprop'
          , z_dim=100
          )

if mode == 'build':
    pass
    # gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

gan.discriminator.summary()

gan.generator.summary()

# training

BATCH_SIZE = 64
EPOCHS = 6000
PRINT_EVERY_N_BATCHES = 5

gan.train(
    x_train
    , batch_size=BATCH_SIZE
    , epochs=EPOCHS
    , run_folder=RUN_FOLDER
    , print_every_n_batches=PRINT_EVERY_N_BATCHES
)

fig = plt.figure()
plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

plt.plot([x[1] for x in gan.d_losses], color='green', linewidth=0.25)
plt.plot([x[2] for x in gan.d_losses], color='red', linewidth=0.25)
plt.plot([x[0] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.xlim(0, 2000)
plt.ylim(0, 2)

plt.show()

fig = plt.figure()
plt.plot([x[3] for x in gan.d_losses], color='black', linewidth=0.25)
plt.plot([x[4] for x in gan.d_losses], color='green', linewidth=0.25)
plt.plot([x[5] for x in gan.d_losses], color='red', linewidth=0.25)
plt.plot([x[1] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('accuracy', fontsize=16)

plt.xlim(0, 2000)

plt.show()
