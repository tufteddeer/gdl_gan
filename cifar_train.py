import os
import numpy as np
import matplotlib.pyplot as plt

from WGAN import WGAN
from loaders import load_cifar

# run params
SECTION = 'gan'
RUN_ID = '0002'
DATA_NAME = 'horses'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode = 'build'  # 'load' #

# 1: Cars, 7: Horses
(x_train, y_train) = load_cifar(7, 10)

plt.imshow((x_train[150, :, :, :] + 1) / 2)

gan = WGAN(input_dim=(32, 32, 3),
           critic_conv_filters=[32, 64, 128, 128],
           critic_conv_kernel_size=[5, 5, 5, 5],
           critic_conv_strides=[2, 2, 2, 1],
           critic_batch_norm_momentum=None,
           critic_activation='leaky_relu',
           critic_dropout_rate=None,
           critic_learning_rate=0.00005,
           generator_initial_dense_layer_size=(4, 4, 128),
           generator_upsample=[2, 2, 2, 1],
           generator_conv_filters=[128, 64, 32, 3],
           generator_conv_kernel_size=[5, 5, 5, 5],
           generator_conv_strides=[1, 1, 1, 1],
           generator_batch_norm_momentum=0.8,
           generator_activation='leaky_relu',
           generator_dropout_rate=None,
           generator_learning_rate=0.00005,
           optimiser='rmsprop',
           z_dim=100)

if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

gan.critic.summary()

gan.generator.summary()

BATCH_SIZE = 128
EPOCHS = 6000
PRINT_EVERY_N_BATCHES = 5
N_CRITIC = 5
CLIP_THRESHOLD = 0.01

gan.train(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER,
    print_every_n_batches=PRINT_EVERY_N_BATCHES,
    n_critic=N_CRITIC,
    clip_threshold=CLIP_THRESHOLD
)

gan.sample_images(RUN_FOLDER)

fig = plt.figure()
plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

plt.plot([x[1] for x in gan.d_losses], color='green', linewidth=0.25)
plt.plot([x[2] for x in gan.d_losses], color='red', linewidth=0.25)
plt.plot(gan.g_losses, color='orange', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('loss', fontsize=16)

# plt.xlim(0, 2000)
# plt.ylim(0, 2)

plt.show()


def compare_images(img1, img2):
    return np.mean(np.abs(img1 - img2))


r, c = 5, 5

idx = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
true_imgs = (x_train[idx] + 1) * 0.5

fig, axs = plt.subplots(r, c, figsize=(15, 15))
cnt = 0

for i in range(r):
    for j in range(c):
        axs[i, j].imshow(true_imgs[cnt], cmap='gray_r')
        axs[i, j].axis('off')
        cnt += 1
fig.savefig(os.path.join(RUN_FOLDER, "images/real.png"))
plt.close()

r, c = 5, 5
noise = np.random.normal(0, 1, (r * c, gan.z_dim))
gen_imgs = gan.generator.predict(noise)

# Rescale images 0 - 1

gen_imgs = 0.5 * (gen_imgs + 1)
# gen_imgs = np.clip(gen_imgs, 0, 1)

fig, axs = plt.subplots(r, c, figsize=(15, 15))
cnt = 0

for i in range(r):
    for j in range(c):
        axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray_r')
        axs[i, j].axis('off')
        cnt += 1
fig.savefig(os.path.join(RUN_FOLDER, "images/sample.png"))
plt.close()

fig, axs = plt.subplots(r, c, figsize=(15, 15))
cnt = 0

for i in range(r):
    for j in range(c):
        c_diff = 99999
        c_img = None
        for k_idx, k in enumerate((x_train + 1) * 0.5):

            diff = compare_images(gen_imgs[cnt, :, :, :], k)
            if diff < c_diff:
                c_img = np.copy(k)
                c_diff = diff
        axs[i, j].imshow(c_img, cmap='gray_r')
        axs[i, j].axis('off')
        cnt += 1

fig.savefig(os.path.join(RUN_FOLDER, "images/sample_closest.png"))
plt.close()
