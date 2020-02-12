from skimage.io import imsave, imread
from tqdm import tqdm
import os
import sys
import numpy as np

image_path = "/home/amin/Semantic_Segmentation/Ours/2/keras-semantic-segmentation-amin-eric-master/Resized/1024x1024/Resized_806_Label_1024x1024_HQ/"
output_path = "/home/amin/Semantic_Segmentation/Ours/2/keras-semantic-segmentation-amin-eric-master/Resized/1024x1024/Resized_806_Label_1024x1024_RGB_Sharp/"
image_files = next(os.walk(image_path))[2]
image_rows, image_cols = 1024, 1024

CLASS_MAPPINGS = [
    {
        'name': 'Necrosis',
        'ranges': [
            (3, 6),
            (3, 6),
            (3, 6),
        ],
        'target': (5, 5, 5)
    },
    {
        'name': 'Cellular Tumor',
        'ranges': [
            (0, 100),
            (100, 255),
            (0, 80),
        ],
        'target': (5, 208, 4)
    },
    {
        'name': 'Leading Edge',
        'ranges': [
            (0, 100),
            (43, 173),
            (90, 255),
        ],
        'target': (33, 143, 166)
    },
    {
        'name': 'Cellular Tumor',
        'ranges': [
            (0, 66),
            (174, 255),
            (150, 220),
        ],
        'target': (6, 208, 170)
    },
    {
        'name': 'Infiltrating Tumor',
        'ranges': [
            (160, 255),
            (0, 65),
            (150, 255),
        ],
        'target': (210, 5, 208)
    },
    {
        'name': '???',
        'ranges': [
            (200, 255),
            (50, 150),
            (0, 30),
        ],
        'target': (255, 102, 0)
    },
    {
        'name': '???',
        'ranges': [
            (0, 85),
            (175, 255),
            (200, 255),
        ],
        'target': (37, 209, 247)
    },
    {
        'name': '???',
        'ranges': [
            (0, 85),
            (175, 255),
            (200, 255),
        ],
        'target': (37, 209, 247)
    },
    {
        'name': 'Blood Cells',
        'ranges': [
            (200, 255),
            (200, 255),
            (200, 255),
        ],
        'target': (255, 255, 255)
    },
]

sys.stdout.flush()
for x, file_name in tqdm(enumerate(image_files), total=len(image_files)):

    remapped_image = np.zeros((image_rows, image_cols, 3), dtype="uint8")
    raw_image = imread(image_path + file_name)

    for i in range(image_rows):
        for j in range(image_cols):
            class_matched = False
            for class_mapping in CLASS_MAPPINGS:
                name = class_mapping['name']
                target = class_mapping['target']
                matches_class = True
                for c, (channel_min, channel_max) in enumerate(class_mapping['ranges']):
                    matches_class &= channel_min <= raw_image[i, j, c] <= channel_max
                if matches_class:
                    remapped_image[i, j] = target
                    class_matched = True
                    break
            if not class_matched and (i != 0) & (j != 0):
                remapped_image[i, j] = remapped_image[i, j - 1]

    name, ext = os.path.splitext(file_name)
    file_path_name = os.path.join(output_path, name + ".png")
    imsave(file_path_name, remapped_image)