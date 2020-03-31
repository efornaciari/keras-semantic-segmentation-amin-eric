import os
import argparse
from tqdm import tqdm
from skimage.io import imread, imsave
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

DEFAULT_PATCH_SIZE = 256


def run(
        input_path,
        output_path,
        patch_size,
):
    num_images = next(os.walk(input_path))[2]
    for index, file_name in tqdm(enumerate(num_images), total=len(num_images)):
        image = imread(input_path + file_name)

        height, width = image.shape[0], image.shape[1]

        num_patches_in_rows = int(height / patch_size)

        mun_patches_in_cols = int(width / patch_size)
        name, ext = os.path.splitext(file_name)

        counter = 0
        for i in range(0, mun_patches_in_cols):
            step_rows = patch_size * i

            for j in range(0, num_patches_in_rows):
                step_cols = patch_size * j
                patch = image[step_rows: step_rows + patch_size, step_cols: step_cols + patch_size, :]
                file_name = '{name}_{patch_id}.{ext}'.format(name=name, patch_id=str(counter).zfill(2), ext='png')
                file_name = os.path.join(output_path, file_name)
                imsave(file_name, patch)
                counter += 1


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('Extract full patches')
    arg_parser.add_argument("--input-path", required=True)
    arg_parser.add_argument("--output-path", required=True)
    arg_parser.add_argument("--patch-size", default=DEFAULT_PATCH_SIZE)
    args = arg_parser.parse_args()

    run(input_path=args.input_path,
        output_path=args.output_path,
        patch_size=args.patch_size)
