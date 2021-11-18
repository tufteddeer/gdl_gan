import numpy as np
from matplotlib import pyplot as plt

from loaders import load_safari

# draw some random samples from the camel data
if __name__ == '__main__':

    images = load_safari("../data/camel/camel.npy")
    np.random.shuffle(images)
    rows = columns = 5
    fig, axs = plt.subplots(rows, columns, figsize=(15, 15))
    cnt = 0
    for i in range(rows):
        for j in range(columns):
            axs[i, j].imshow((images[cnt, :, :, :]), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()
    fig.savefig("camel_samples.eps")
