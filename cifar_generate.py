import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

from WGAN import WGAN
import loaders

# source of randomness
rng = default_rng()

model_file = "weights-135.h5"
model: WGAN = loaders.load_model(WGAN, 'run/gan/0002_horses', model_file)


def gen_image(input_noise):
    img = model.generator.predict(input_noise)[0]
    # Rescale images 0 - 1
    img = 0.5 * (img + 1)
    img = np.clip(img, 0, 1)
    return img


if __name__ == '__main__':

    input_shape = (1, model.z_dim)
    noise = rng.standard_normal(input_shape)
    vec = rng.integers(0, 2, input_shape) * 0.5

    plt.imshow(gen_image(noise))
    plt.show()
