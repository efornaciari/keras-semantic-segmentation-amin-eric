import os
import model
import numpy as np
from skimage.io import imread
from keras.callbacks import EarlyStopping, ModelCheckpoint


def main():
    images_path = '/Users/ericfornaciari/Desktop/data/images/patches/raw/1024x1024/'
    masks_path = '/Users/ericfornaciari/Desktop/data/masks/patches/encoded/1024x1024/'

    images = np.zeros((len(os.listdir(images_path)), 1024, 1024, 3), np.uint16)
    masks = np.zeros((len(os.listdir(masks_path)), 1024, 1024, 1), np.bool)
    for i, filename in enumerate(os.listdir(images_path)):
        images[i, :, :, :] = imread(os.path.join(images_path, filename))
    for i, filename in enumerate(os.listdir(masks_path)):
        masks[i, :, :, :] = np.load(os.path.join(masks_path, filename), allow_pickle=True)

    unet = model.unet(input_size=(1024, 1024, 3))
    unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('init.h5', verbose=1, save_best_only=True)
    unet.fit(images, masks, validation_split=0.1, batch_size=16, epochs=50, callbacks=[earlystopper, checkpointer])


if __name__ == '__main__':
    main()
