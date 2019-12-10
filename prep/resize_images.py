from os.path import isfile, join, splitext

import os

from PIL import Image

indir = '../washingtondb-v1.0/data/word_images_normalized/'
outdir = 'generated-images/'
target_dimensions = (160, 30)

images = [f for f in os.listdir(indir) if isfile(join(indir, f)) and splitext(f)[1] == '.png']

for img_path in images[:3]:
    img = Image.open(join(indir, img_path))
    ratio = target_dimensions[1] / img.size[1]
    new_dimensions = tuple([int(dim * ratio) for dim in img.size])
    resized_img = img.resize(new_dimensions)  # TODO
    padded_img = Image.new('RGB', target_dimensions, (0, 0, 0))
    padded_img.paste(resized_img, ((target_dimensions[0] - new_dimensions[0]) // 2,
                                   (target_dimensions[1] - new_dimensions[1]) // 2))

    # padded_img.show()
    new_img_path = join(outdir, img_path)
    with open(new_img_path, 'wb+') as img_file:
        padded_img.save(img_file, format='PNG')
