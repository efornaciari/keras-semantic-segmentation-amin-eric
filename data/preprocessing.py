import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000


def one_hot_encode_rgb_masks(masks, encodings):
    """
    [
        {
            "name": "Background",
            "color": (255, 255, 255)
        },
        {
            "name": "Leading Edge",
            "color": (33, 143, 166)
        },
        {
            "name": "Infiltrating Tumor",
            "color": (1, 2, 3)
        },
        {
            "name": "Cellular Tumor",
            "color": (1, 2, 3)
        },
        {
            "name": "Perinecrotic Zone",
            "color": (1, 2, 3)
        },
        {
            "name": "Necrosis",
            "color": (1, 2, 3)
        }
    ]

    :param masks: 
    :param encodings: 
    :return: 
    """

    num_classes = len(encodings)
    num_images, height, width = masks.shape[:-1]

    one_hot_encoded_rgb_images = np.zeros((num_images, height, width, num_classes), dtype=np.bool)

    for i in range(num_images):
        mask = masks[i, :, :, :]
        for j in range(len(encodings)):
            encoding = encodings[j]
            color = encoding["color"]
            one_hot_encoded_rgb_images[i, :, :, j] = np.all(mask == color, axis=-1)
    return one_hot_encoded_rgb_images
