import os
import argparse
import numpy as np
from skimage.io import imread

import data.preprocessing as preprocessing

DEFAULT_INPUT_FOLDER = 'masks/raw'
DEFAULT_OUTPUT_FOLDER = 'masks/encoded'

COPY = 'copy'
RENAME = 'rename'

ENCODINGS = [
    {
        "name": "Blood Cells",
        "color": (255, 255, 255)
    },
    {
        "name": "Leading Edge",
        "color": (33, 143, 166)
    },
    {
        "name": "Infiltrating Tumor",
        "color": (210, 5, 208)
    },
    {
        "name": "Cellular Tumor",
        "color": (5, 208, 4)
    },
    {
        "name": "Perinecrotic Zone",
        "color": (67, 209, 247)
    },
    {
        "name": "Necrosis",
        "color": (5, 5, 5)
    }
]


def run(
        base_directory,
        input_folder,
        output_folder,
        dry_run,
        verbose
):
    input_directory = os.path.join(base_directory, input_folder)
    output_directory = os.path.join(base_directory, output_folder)

    if not os.path.exists(input_directory):
        raise Exception('Input directory must exist: "{input_directory}"'
                        .format(input_directory=input_directory))

    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    for mask_raw_filename in os.listdir(input_directory):
        mask_filename_without_ext, _ = os.path.splitext(mask_raw_filename)

        mask_encoded_filename = '{mask_filename_without_ext}.np' \
            .format(mask_filename_without_ext=mask_filename_without_ext)

        src_mask_raw_filename = os.path.join(input_directory, mask_raw_filename)
        dst_mask_raw_filename = os.path.join(output_directory, mask_encoded_filename)

        if verbose:
            print('Converting mask "{src_mask_raw_filename}" to "{dst_mask_raw_filename}"'.format(
                src_mask_raw_filename=src_mask_raw_filename,
                dst_mask_raw_filename=dst_mask_raw_filename))

        if not dry_run:
            raw_masks = np.expand_dims(imread(src_mask_raw_filename), axis=0)
            encoded_mask = preprocessing.one_hot_encode_rgb_masks(masks=raw_masks, encodings=ENCODINGS)
            np.save(dst_mask_raw_filename, encoded_mask)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('Encodes raw mask images')
    arg_parser.add_argument("--base-directory", required=True)
    arg_parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER)
    arg_parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)
    arg_parser.add_argument("--dry-run", action='store_true')
    arg_parser.add_argument("--verbose", action='store_true')

    args = arg_parser.parse_args()

    run(base_directory=args.base_directory,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        dry_run=args.dry_run,
        verbose=args.verbose)
