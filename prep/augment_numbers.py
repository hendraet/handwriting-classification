import json

import copy
import os

import numpy as np
np.random.bit_generator = np.random._bit_generator

from PIL import Image
from imgaug import augmenters as iaa
from resize_images import resize_img

IN_JSON = 'dataset_descriptions/iamondb_generated_dates.json'
OUT_JSON = 'dataset_descriptions/iamondb_generated_dates_aug.json'
IMG_IN_DIR = 'datasets/tars'
IMG_OUT_DIR = 'datasets/'
SCALE_FACTOR = 1  # How much bigger the new dataset should be
SAVE_ORIGINAL = False
TARGET_DIMENSIONS = (350, 60)

new_image_infos = []

with open(IN_JSON, 'r') as json_file:
    json_images = json.load(json_file)

image_infos = copy.deepcopy(json_images)
for idx in range(0, len(image_infos)):
    image_infos[idx]['path'] = os.path.join(IMG_IN_DIR, os.path.basename(image_infos[idx]['path']))

seq = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # Small blur on half of the images
    iaa.ContrastNormalization((0.75, 1.5)),  # Change in contrast
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # Gaussian Noise can change colour
    iaa.Multiply((0.8, 1.2), per_channel=0.2)#,  # brighter/darker
    # iaa.Affine(
    #     rotate=(-10, 10),
    #     shear=(-20, 20),
    #     fit_output=True,
    # ),
    # iaa.PiecewiseAffine(scale=(0.01, 0.05), mode='edge')
], random_order=True)  # apply augmenters in random order

images = np.array([np.array(Image.open(img['path'])) for img in image_infos])

images = np.repeat(images, SCALE_FACTOR, axis=0)
image_infos = np.repeat(np.array(image_infos), SCALE_FACTOR, axis=0)

# Make batches out of single images because processing multiple images at once leads to errors because the resulting
# images have different shapes and therefore can't be organised in a numpy array -> leads to an internal error
for idx, img in enumerate(images):
    old_filename = os.path.splitext(os.path.basename(image_infos[idx]['path']))[0]
    new_filename = old_filename + '-' + str(idx % SCALE_FACTOR + 1) + '.png'
    new_img_path = os.path.join(IMG_OUT_DIR, new_filename)

    if SAVE_ORIGINAL and idx % SCALE_FACTOR == 0:
        with open(os.path.join(IMG_OUT_DIR, old_filename + '.png'), 'wb+') as orig_img_file:
            resize_img(Image.fromarray(img), TARGET_DIMENSIONS).save(orig_img_file, format='PNG')

    img_aug = seq(images=np.expand_dims(img, axis=0))
    pil_img = Image.fromarray(np.squeeze(img_aug), 'L')
    padded_img = resize_img(pil_img, TARGET_DIMENSIONS)

    with open(new_img_path, 'wb+') as img_file:
        padded_img.save(img_file, format='PNG')

    new_image_infos.append({
        "string": image_infos[idx]['string'],
        "type": image_infos[idx]['type'],
        "path": new_img_path
    })

with open(OUT_JSON, 'w') as out_json:
    if SAVE_ORIGINAL:
        assert os.path.split(json_images[0]['path'])[0] == os.path.split(new_image_infos[0]['path'])[0],\
            "Original path differs from new path"
        json.dump(json_images + new_image_infos, out_json, indent=4)
    else:
        json.dump(new_image_infos, out_json, indent=4)

print(f'Created {len(new_image_infos)} new images (original: {len(json_images)})')
