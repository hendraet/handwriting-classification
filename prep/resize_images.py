import json
import os

from PIL import Image
from os.path import join


# Useful for recursive traversion:
# tmp = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(indir)) for f in fn]

def resize_img(img, target_dimensions, padding_color=0):
    ratio = target_dimensions[1] / img.size[1]
    new_dimensions = tuple([int(dim * ratio) for dim in img.size])

    if new_dimensions[0] > target_dimensions[0]:
        ratio = target_dimensions[0] / img.size[0]
        new_dimensions = tuple([int(dim * ratio) for dim in img.size])
    resized_img = img.resize(new_dimensions)

    padded_img = Image.new('L', target_dimensions, color=padding_color)
    # center image vertically but pad only to the right
    padded_img.paste(resized_img, (0, (target_dimensions[1] - new_dimensions[1]) // 2))

    return padded_img


def main():
    outdir = 'datasets/tars'
    indir = 'datasets/'
    json_path = 'dataset_descriptions/iamondb_generated_nums_20k.json'
    # json_path = 'datasets/tars/iamondb_generated_dates_50k.json'
    target_dimensions = (216, 64)
    padding_colour = 255

    with open(json_path, 'r') as j_file:
        num_json = json.load(j_file)

    images = [i['path'] for i in num_json]
    for img_path in images:
        img_path = os.path.basename(img_path)
        img = Image.open(join(indir, img_path))
        padded_img = resize_img(img, target_dimensions, padding_colour)

        # padded_img.show()
        new_img_path = join(outdir, img_path)
        with open(new_img_path, 'wb+') as img_file:
            padded_img.save(img_file, format='PNG')


if __name__ == '__main__':
    main()
