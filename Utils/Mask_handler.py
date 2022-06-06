from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image


def visualize_mask(mask):
    plt.imshow(mask, cmap='Greys')
    plt.show()


def visualize_photo(photo):
    plt.imshow(photo)
    plt.show()


def reshape_image(image, new_dim=(1920, 1080)):
    """
    Function to reshape the image to any dimension

    Args:
        image (image): number of nodes (instances) at each frame
        new_dim (tuple): choose if plot or not the graph once created
    Returns:
        new_image (image): resized image to the desired dimensions
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print('Original Dimensions : ', image.shape)
    new_image = image[:][224:(1024 - 224)]
    new_image = cv2.resize(new_image, new_dim)
    return new_image


def reshape_mask(mask, new_dim=(1920, 1080)):
    # print('Original Dimensions : ', mask.shape)
    new_mask = mask[:][224:(1024 - 224)]
    new_mask = cv2.resize(np.uint8(new_mask), new_dim)
    # print('New Dimensions : ', new_mask.shape)
    return new_mask
