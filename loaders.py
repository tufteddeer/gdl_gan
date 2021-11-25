import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar100, cifar10


def load_safari(path):
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


def load_cifar(label, num):
    if num == 10:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    train_mask = [y[0] == label for y in y_train]
    test_mask = [y[0] == label for y in y_test]

    x_data = np.concatenate([x_train[train_mask], x_test[test_mask]])
    y_data = np.concatenate([y_train[train_mask], y_test[test_mask]])

    x_data = (x_data.astype('float32') - 127.5) / 127.5

    return x_data, y_data
