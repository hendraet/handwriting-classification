import json
import tarfile

import random

import os
import shutil

from PIL import Image
from os.path import join


# Useful for recursive traversion:
# tmp = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(indir)) for f in fn]

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
    tar_dir = "datasets/tars"
    outdir = "../handwriting_embedding/datasets"
    indir = "datasets/"
    dataset_description = "dataset_descriptions/mixed_gw_rep_words_dates_40k.json"
    new_name = "mixed_gw_rep_words_dates_40k_padded"
    # dataset_description = "datasets/tars/iamondb_generated_dates_50k.json"
    target_dimensions = (216, 64)
    padding_colour = 255
    additional_padding = True

    image_files = [fn for fn in os.listdir(outdir) if fn.endswith(".png")]
    # TODO
    # assert not image_files, "There are already image files in the outdir"

    with open(dataset_description, "r") as j_file:
        num_json = json.load(j_file)

    # TODO
    # images = [i["path"] for i in num_json]
    images = [i["path"] for i in num_json if i["type"] == "date"]
    for img_path in images:
        img_path = os.path.basename(img_path)
        img = Image.open(join(indir, img_path))

        padding_right = max(0, int(random.gauss(0, 35))) if additional_padding else 0
        padded_img = resize_img(img, target_dimensions, padding_colour, additional_padding=(0, 0, padding_right, 0))

        new_img_path = join(outdir, img_path)
        with open(new_img_path, "wb") as img_file:
            padded_img.save(img_file, format="PNG")

    # copy description
    new_dataset_description = os.path.join(outdir, new_name + ".json")
    shutil.copy(dataset_description, new_dataset_description)

    # make tar
    tar_filename = os.path.join(tar_dir, new_name + ".tar.bz2")
    image_files = [os.path.join(outdir, fn) for fn in os.listdir(outdir) if fn.endswith(".png")]
    with tarfile.open(tar_filename, "w:bz2") as tar:
        for filename in image_files:
            tar.add(filename, arcname=os.path.basename(filename))
        tar.add(new_dataset_description, arcname=os.path.basename(new_dataset_description))


if __name__ == "__main__":
    main()
