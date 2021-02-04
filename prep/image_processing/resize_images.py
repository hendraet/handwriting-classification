import argparse
import json
from os.path import join

import os
import random
import shutil
from PIL import Image

from prep.utils import create_tar


# additional padding: (left, top, right, bot)
def resize_img(img, target_dimensions, padding_color=0, additional_padding=(0, 0, 0, 0)):
    # adapt target dimensions
    new_width = target_dimensions[0] - additional_padding[0] - additional_padding[2]
    new_height = target_dimensions[1] - additional_padding[1] - additional_padding[3]
    padded_dimensions = (new_width, new_height)

    ratio = padded_dimensions[1] / img.size[1]
    new_dimensions = tuple([int(dim * ratio) for dim in img.size])

    if new_dimensions[0] > padded_dimensions[0]:
        ratio = padded_dimensions[0] / img.size[0]
        new_dimensions = tuple([int(dim * ratio) for dim in img.size])
    resized_img = img.resize(new_dimensions)

    padded_img = Image.new("L", target_dimensions, color=padding_color)
    # center image vertically but pad only to the right
    top_left_x = additional_padding[0]
    top_left_y = (padded_dimensions[1] - new_dimensions[1]) // 2 + additional_padding[1]
    padded_img.paste(resized_img, (top_left_x, top_left_y))

    return padded_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("orig_dataset_dir", type=str, help="the directory of the original dataset")
    parser.add_argument("dataset_dir", type=str, help="the directory where the resulting dataset should be saved to")
    parser.add_argument("new_dataset_name", type=str, help="the name of the new dataset")
    parser.add_argument("--target-dimensions", nargs=2, default=[216, 64],
                        help="new dimensions of the images in (w x h)")
    parser.add_argument("--tar-dir", type=str, default="datasets/tars",
                        help="the directory where the resulting tar archive should be stored")
    parser.add_argument("--padding_colour", type=int, default=255)
    parser.add_argument("--additional-padding", action="store_true", default=False)
    args = parser.parse_args()

    orig_dataset_dir = args.orig_dataset_dir
    dataset_dir = args.dataset_dir
    tar_dir = args.tar_dir
    new_dataset_name = args.new_dataset_name
    target_dimensions = args.target_dimensions
    padding_colour = args.padding_colour
    additional_padding = args.additional_padding

    possible_descriptions = [filename for filename in os.listdir(orig_dataset_dir) if filename.endswith(".json")]
    assert len(possible_descriptions) == 1, "There should be exactly one json file in the original dataset directory"
    dataset_description_filename = os.path.join(orig_dataset_dir, possible_descriptions[0])

    image_files = [fn for fn in os.listdir(dataset_dir) if fn.endswith(".png")]
    assert not image_files, "There are already image files in the out dir"

    with open(dataset_description_filename, "r") as j_file:
        num_json = json.load(j_file)

    images = [i["path"] for i in num_json]
    for img_path in images:
        img_path = os.path.basename(img_path)
        img = Image.open(join(orig_dataset_dir, img_path))
        import numpy
        if 0 in numpy.array(img):
            print(img_path)

        padding_right = max(0, int(random.gauss(0, 35))) if additional_padding else 0
        padded_img = resize_img(img, target_dimensions, padding_colour, additional_padding=(0, 0, padding_right, 0))

        new_img_path = join(dataset_dir, img_path)
        with open(new_img_path, "wb") as img_file:
            padded_img.save(img_file, format="PNG")

    # copy description
    new_dataset_description = os.path.join(dataset_dir, new_dataset_name + ".json")
    shutil.copy(dataset_description_filename, new_dataset_description)

    create_tar(new_dataset_name, new_dataset_description, dataset_dir, tar_dir)


if __name__ == "__main__":
    main()
