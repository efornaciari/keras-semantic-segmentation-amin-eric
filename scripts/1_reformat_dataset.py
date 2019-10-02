import os
import argparse
import fnmatch
from shutil import copyfile

MASK_PATTERN = '*_GT.jpg'
DEFAULT_IMAGES_FOLDER = 'images'
DEFAULT_MASKS_FOLDER = 'masks'

COPY = 'copy'
RENAME = 'rename'


def normalize_filename_from_mask_filename(mask_filename):
    mask_filename_without_extension, mask_filename_extension = os.path.splitext(mask_filename)
    return "{mask_filename_without_extension}{mask_filename_extension}".format(
        mask_filename_without_extension=mask_filename_without_extension.replace('_GT', ''),
        mask_filename_extension=mask_filename_extension)


def run(
        input_directory,
        output_directory,
        images_folder,
        masks_folder,
        action,
        dry_run,
        verbose
):
    if not os.path.exists(input_directory):
        raise Exception('Input directory must exist: "{input_directory}"'
                        .format(input_directory=input_directory))
    if not os.path.exists(output_directory):
        if verbose:
            print('Creating output directory "{output_directory}"'
                  .format(output_directory=output_directory))
        if not dry_run:
            os.makedirs(output_directory, exist_ok=True)

    images_directory = os.path.join(output_directory, images_folder)
    masks_directory = os.path.join(output_directory, masks_folder)

    if not dry_run:
        os.makedirs(images_directory, exist_ok=True)
        os.makedirs(masks_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if fnmatch.fnmatch(filename, MASK_PATTERN):
            mask_filename = filename
            image_filename = normalize_filename_from_mask_filename(mask_filename)
            src_image_filename = os.path.join(input_directory, image_filename)
            src_mask_filename = os.path.join(input_directory, mask_filename)
            dst_image_filename = os.path.join(images_directory, image_filename)
            dst_mask_filename = os.path.join(masks_directory, image_filename)

            if action == RENAME:
                if verbose:
                    print('Renaming image: "{src_image_filename}" to "{dst_image_filename}"'.format(
                        src_image_filename=src_image_filename,
                        dst_image_filename=dst_image_filename))
                    print('Renaming mask: "{src_mask_filename}" to "{dst_mask_filename}"'.format(
                        src_mask_filename=src_mask_filename,
                        dst_mask_filename=dst_mask_filename))
                if not dry_run:
                    os.rename(src_image_filename, dst_image_filename)
                    os.rename(src_mask_filename, dst_mask_filename)
            if action == COPY:
                if verbose:
                    print('Copying image: "{src_image_filename}" to "{dst_image_filename}"'.format(
                        src_image_filename=src_image_filename,
                        dst_image_filename=dst_image_filename))
                    print('Copying mask: "{src_mask_filename}" to "{dst_mask_filename}"'.format(
                        src_mask_filename=src_mask_filename,
                        dst_mask_filename=dst_mask_filename))
                if not dry_run:
                    copyfile(src_image_filename, dst_image_filename)
                    copyfile(src_mask_filename, dst_mask_filename)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('Restructures Dataset to be divided into images & masks')
    arg_parser.add_argument("--input-directory", required=True)
    arg_parser.add_argument("--output-directory", required=True)
    arg_parser.add_argument("--images-folder", default=DEFAULT_IMAGES_FOLDER)
    arg_parser.add_argument("--masks-folder", default=DEFAULT_MASKS_FOLDER)
    arg_parser.add_argument("--action", choices=[COPY, RENAME], required=True)
    arg_parser.add_argument("--dry-run", action='store_true')
    arg_parser.add_argument("--verbose", action='store_true')

    args = arg_parser.parse_args()

    run(input_directory=args.input_directory,
        output_directory=args.output_directory,
        images_folder=args.images_folder,
        masks_folder=args.masks_folder,
        action=args.action,
        dry_run=args.dry_run,
        verbose=args.verbose)
