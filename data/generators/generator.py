import os
import numpy as np
from keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize as skimage_resize

from data import preprocessing


class Generator(Sequence):
    def __init__(self, images_directory, masks_directory, encodings, batch_size, resize=None):
        resize = self._format_resize(resize)
        self.images = self._build_np_array_from_directory(images_directory, resize=resize)
        self.masks = self._build_np_array_from_directory(masks_directory, resize=resize)
        self.batch_size = batch_size
        self.encodings = encodings

    def __len__(self):
        return np.ceil(len(self.images) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = np.array(self.images[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_y = self._build_mask(np.array(self.masks[idx * self.batch_size:(idx + 1) * self.batch_size]))
        return batch_x, batch_y

    def _build_mask(self, masks):
        return preprocessing.one_hot_encode_rgb_masks(masks, self.encodings)

    @staticmethod
    def _build_np_array_from_directory(directory, resize=None):
        images = []
        for idx, file in enumerate(os.listdir(directory)):
            print(os.path.join(directory, file))
            image = imread(os.path.join(directory, file))
            if resize is not None:
                image = skimage_resize(image, resize)
            images.append(image)
        return images

    @staticmethod
    def _format_resize(resize):
        if resize is not None:
            if isinstance(resize, int):
                resize = (resize, resize)
            elif isinstance(resize, tuple):
                resize = resize
        return resize
