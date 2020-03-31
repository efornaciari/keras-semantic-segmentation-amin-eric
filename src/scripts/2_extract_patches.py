import os
import random
import fnmatch
import argparse
from skimage.io import imread, imsave
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

DEFAULT_INPUT_IMAGE_FOLDER = 'images/raw'
DEFAULT_INPUT_MASK_FOLDER = 'masks/raw'
DEFAULT_OUTPUT_IMAGE_FOLDER = 'images/patches/raw'
DEFAULT_OUTPUT_MASK_FOLDER = 'masks/patches/raw'
DEFAULT_PATCHES_PER_IMAGE = 5
DEFAULT_PATCH_SIZE = 2048
PATTERN = '*.jpg'

COPY = 'copy'
RENAME = 'rename'


def run(
        base_directory,
        input_image_folder,
        input_mask_folder,
        output_image_folder,
        output_mask_folder,
        patch_size,
        patches_per_image,
        seed,
        dry_run,
        verbose
):
    random.seed(seed)
    output_image_folder = os.path.join(output_image_folder, "{patch_size}x{patch_size}".format(patch_size=patch_size))
    output_mask_folder = os.path.join(output_mask_folder, "{patch_size}x{patch_size}".format(patch_size=patch_size))
    input_image_directory = os.path.join(base_directory, input_image_folder)
    input_mask_directory = os.path.join(base_directory, input_mask_folder)
    output_image_directory = os.path.join(base_directory, output_image_folder)
    output_mask_directory = os.path.join(base_directory, output_mask_folder)

    if not os.path.exists(input_image_directory):
        raise Exception('Input image directory must exist: "{input_image_directory}"'
                        .format(input_image_directory=input_image_directory))
    if not os.path.exists(input_mask_directory):
        raise Exception('Input image directory must exist: "{input_mask_directory}"'
                        .format(input_mask_directory=input_mask_directory))

    if not dry_run:
        os.makedirs(output_image_directory, exist_ok=True)
    if not dry_run:
        os.makedirs(output_mask_directory, exist_ok=True)

    for filename in os.listdir(input_mask_directory):
        if fnmatch.fnmatch(filename, PATTERN):
            filename_without_ext, filename_ext = os.path.splitext(filename)
            src_image_raw_filename = os.path.join(input_image_directory, filename)
            src_mask_raw_filename = os.path.join(input_mask_directory, filename)

            if verbose:
                print('Creating {patches_per_image} patches from "{src_mask_raw_filename}"'.format(
                    patches_per_image=patches_per_image,
                    src_mask_raw_filename=src_mask_raw_filename))
            if not dry_run:
                mask = imread(src_mask_raw_filename)
                image = imread(src_image_raw_filename)
                for patch_number in range(patches_per_image):
                    image_patch, mask_patch = _create_patch(
                        image=image,
                        mask=mask,
                        patch_size=patch_size)
                    image_patch_filename = '{filename_without_ext}_{patch_number}{mask_filename_ext}'.format(
                        filename_without_ext=filename_without_ext,
                        patch_number=str(patch_number).zfill(5),
                        mask_filename_ext=filename_ext)
                    mask_patch_filename = '{filename_without_ext}_{patch_number}{mask_filename_ext}'.format(
                        filename_without_ext=filename_without_ext,
                        patch_number=str(patch_number).zfill(5),
                        mask_filename_ext=filename_ext)
                    dst_image_patch_filename = os.path.join(output_image_directory, image_patch_filename)
                    dst_mask_patch_filename = os.path.join(output_mask_directory, mask_patch_filename)
                    imsave(dst_image_patch_filename, image_patch)
                    imsave(dst_mask_patch_filename, mask_patch)


def _create_patch(image, mask, patch_size):
    height, width, _ = image.shape
    rand_height = random.randint(0, height - patch_size)
    rand_width = random.randint(0, width - patch_size)

    image_patch = image[rand_height:rand_height + patch_size, rand_width:rand_width + patch_size, :]
    mask_patch = mask[rand_height:rand_height + patch_size, rand_width:rand_width + patch_size, :]
    return image_patch, mask_patch


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('Extracts Patches from dataset')
    arg_parser.add_argument("--base-directory", required=True)
    arg_parser.add_argument("--input-image-folder", default=DEFAULT_INPUT_IMAGE_FOLDER)
    arg_parser.add_argument("--input-mask-folder", default=DEFAULT_INPUT_MASK_FOLDER)
    arg_parser.add_argument("--output-image-folder", default=DEFAULT_OUTPUT_IMAGE_FOLDER)
    arg_parser.add_argument("--output-mask-folder", default=DEFAULT_OUTPUT_MASK_FOLDER)
    arg_parser.add_argument("--patch-size", default=DEFAULT_PATCH_SIZE, type=int)
    arg_parser.add_argument("--patches-per-image", default=DEFAULT_PATCHES_PER_IMAGE)
    arg_parser.add_argument("--seed", default=13)
    arg_parser.add_argument("--dry-run", action='store_true')
    arg_parser.add_argument("--verbose", action='store_true')

    args = arg_parser.parse_args()

    run(base_directory=args.base_directory,
        input_image_folder=args.input_image_folder,
        input_mask_folder=args.input_mask_folder,
        output_image_folder=args.output_image_folder,
        output_mask_folder=args.output_mask_folder,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        seed=args.seed,
        dry_run=args.dry_run,
        verbose=args.verbose)
