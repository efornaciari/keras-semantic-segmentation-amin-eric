import os
import model
import numpy as np
from skimage.io import imread
from keras.callbacks import EarlyStopping, ModelCheckpoint


def main():
    images_path = 'path/to/.../project/.../Project_2_Segmentation_reformatted/images/patches/raw/2048x2048/'
    mask_path = 'path/to/.../project/Project_2_Segmentation_reformatted/masks/patches/encoded/2048x2048/'

    images = np.zeros((len(os.listdir(images_path)), 2048, 2048, 3), np.uint16)
    masks = np.zeros((len(os.listdir(mask_path)), 2048, 2048, 6), np.bool)
    for i, filename in enumerate(os.listdir(images_path)):
        images[i, :, :, :] = imread(os.path.join(images_path, filename))
    for i, filename in enumerate(os.listdir(mask_path)):
        masks[i, :, :, :] = np.load(os.path.join(mask_path, filename), allow_pickle=True)

    unet = model.unet()
    unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    unet = model.unet()

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('init.h5', verbose=1, save_best_only=True)
    unet.fit(images, masks, validation_split=0.1, batch_size=16, epochs=50, callbacks=[earlystopper, checkpointer])


if __name__ == '__main__':
    main()
