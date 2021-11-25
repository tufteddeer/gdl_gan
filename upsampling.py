import numpy as np
from tensorflow.keras.layers import UpSampling2D
from matplotlib import pyplot as plt

# Grayscale image: 3x3x1
image = np.array([[1, 0, 1, ],
                  [0, 1, 0, ],
                  [0.5, 0, 0.5, ]])

# Show original image
plt.imshow(image, cmap='gray')
plt.title("original 3x3 image")
plt.show()

# Reshape image: 1 sample, 3 rows, 3 columns, 1 channel
image = image.reshape((1, 3, 3, 1))

# Make prediction with model to get upsampled image
upsampled = UpSampling2D(input_shape=(3, 3, 1))(image).numpy()

# Reshape upsampled image to 6x6
upsampled = upsampled.reshape((6, 6))

# Show upsampled image
plt.imshow(upsampled, cmap='gray')
plt.title("upsampled 6x6 image")
plt.show()
