import fnmatch
import os
import argparse
from skimage.io import imread, imsave
from sklearn.feature_extraction import image
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

DEFAULT_INPUT_FOLDER = 'masks/raw'
DEFAULT_OUTPUT_FOLDER = 'masks/raw/patches'
DEFAULT_PATCHES_PER_IMAGE = 50
DEFAULT_PATCH_SIZE = 2048
PATTERN = '*.jpg'

COPY = 'copy'
RENAME = 'rename'


def run(
        base_directory,
        input_folder,
        output_folder,
        patch_size,
        patches_per_image,
        dry_run,
        verbose
):
    output_folder = os.path.join(output_folder, "{patch_size}x{patch_size}".format(patch_size=patch_size))
    input_directory = os.path.join(base_directory, input_folder)
    output_directory = os.path.join(base_directory, output_folder)

    if not os.path.exists(input_directory):
        raise Exception('Input directory must exist: "{input_directory}"'
                        .format(input_directory=input_directory))

    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    for mask_raw_filename in os.listdir(input_directory):
        if fnmatch.fnmatch(mask_raw_filename, PATTERN):
            mask_filename_without_ext, mask_filename_ext = os.path.splitext(mask_raw_filename)
            src_mask_raw_filename = os.path.join(input_directory, mask_raw_filename)

            if verbose:
                print('Creating {patches_per_image} patches from "{src_mask_raw_filename}"'.format(
                    patches_per_image=patches_per_image,
                    src_mask_raw_filename=src_mask_raw_filename))
            if not dry_run:
                raw_mask = imread(src_mask_raw_filename)
                raw_mask_patches = image.extract_patches_2d(
                    image=raw_mask,
                    patch_size=(patch_size, patch_size),
                    max_patches=0.00001)
                for patch_number in range(patches_per_image):
                    mask_encoded_filename = '{mask_filename_without_ext}_{patch_number}{mask_filename_ext}'.format(
                        mask_filename_without_ext=mask_filename_without_ext,
                        patch_number=str(patch_number).zfill(5),
                        mask_filename_ext=mask_filename_ext)
                    dst_mask_raw_filename = os.path.join(output_directory, mask_encoded_filename)
                    imsave(dst_mask_raw_filename, raw_mask_patches[patch_number, :, :, :])


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('Encodes raw mask images')
    arg_parser.add_argument("--base-directory", required=True)
    arg_parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER)
    arg_parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)
    arg_parser.add_argument("--patch-size", default=DEFAULT_PATCH_SIZE)
    arg_parser.add_argument("--patches-per-image", default=DEFAULT_PATCHES_PER_IMAGE)
    arg_parser.add_argument("--dry-run", action='store_true')
    arg_parser.add_argument("--verbose", action='store_true')

    args = arg_parser.parse_args()

    run(base_directory=args.base_directory,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        dry_run=args.dry_run,
        verbose=args.verbose)
