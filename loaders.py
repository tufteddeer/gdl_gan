import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator


def load_safari():
    path = "data/camel/camel.npy"

    print("Loading camel images from {}".format(path))
    data = np.load(path)
    # normalize data to range [-1, 1]
    data = (data.astype('float32') - 127.5) / 127.5
    data = data.reshape(data.shape[0], 28, 28, 1)

    return data


def load_celeb(data_name, image_size, batch_size):
    data_folder = os.path.join("./data", data_name)

    data_gen = ImageDataGenerator(preprocessing_function=lambda x: (x.astype('float32') - 127.5) / 127.5)

    x_train = data_gen.flow_from_directory(data_folder
                                           , target_size=(image_size, image_size)
                                           , batch_size=batch_size
                                           , shuffle=True
                                           , class_mode='input'
                                           , subset="training"
                                           )

    return x_train
