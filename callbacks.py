import matplotlib.pyplot as plt
import numpy as np
import keras.callbacks as callbacks


def identity(x):
    return x


class WeightVisualizerCallback(callbacks.Callback):
    def __init__(self, images, masks, figsize=None, mask_decoding_func=None):
        super().__init__()
        assert images.shape[:-1] == masks.shape[:-1]
        if isinstance(figsize, int):
            figsize = (figsize, figsize)
        if figsize is None:
            figsize = tuple(images.shape[1:3])
        if mask_decoding_func is None:
            mask_decoding_func = identity
        self.figsize = figsize
        self.images = images
        self.masks = masks
        self.mask_decoding_func = mask_decoding_func

    def on_epoch_end(self, epoch, logs={}):
        predicted_masks = self.mask_decoding_func(self.model.predict(self.images))
        rows = self.images.shape[0]
        cols = 3
        fig, subplots = plt.subplots(rows, cols, figsize=self.figsize)
        fig.suptitle("Visualizing Weights for epoch: {}".format(epoch))
        for subplot, title in zip(subplots[0], ['Images', 'Masks', 'Predicted Masks']):
            subplot.set_title(title)
        for row, row_subplots in enumerate(subplots):
            row_subplots[0].imshow(self.images[row, :, :, :])
            row_subplots[1].imshow(np.squeeze(self.masks[row, :, :, :]))
            row_subplots[2].imshow(np.squeeze(predicted_masks[row, :, :, :]))
        fig.tight_layout()
        plt.show()
