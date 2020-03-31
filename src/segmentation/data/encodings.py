import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000


class OneHotEncoderDecoder:
    def __init__(self, encodings):
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
        """
        self.encodings = encodings
        num_classes = len(self.encodings)
        self.one_hot_to_rbg = {tuple([int(i == j) for j in range(num_classes)]): encoding['color'] for i, encoding in enumerate(encodings)}
        self.rbg_to_one_hot = {encoding['color']: tuple([int(i == j) for j in range(num_classes)]) for i, encoding in enumerate(encodings)}

    def encode(self, images):
        def rgb_to_one_hot(rgb):
            hashable_rgb = tuple(rgb)
            if hashable_rgb not in self.rbg_to_one_hot:
                return [0 for _ in range(len(self.encodings))]
            return self.rbg_to_one_hot[hashable_rgb]
        return np.apply_along_axis(rgb_to_one_hot, axis=-1, arr=images)

    def decode(self, masks):
        def one_hot_to_rgb(one_hot):
            hashable_one_hot = tuple([int(i) for i in one_hot])
            if hashable_one_hot not in self.one_hot_to_rbg:
                return [0, 0, 0]
            return self.one_hot_to_rbg[hashable_one_hot]
        return np.apply_along_axis(one_hot_to_rgb, axis=-1, arr=masks)
