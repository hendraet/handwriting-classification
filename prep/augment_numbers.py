import json
import os

import numpy as np

from PIL import Image
from imgaug import augmenters as iaa
from resize_images import resize_img

IN_JSON = 'dataset_descriptions/numbers_iamdb.json'
IMG_IN_DIR = '../datasets/iamdb/'
IMG_OUT_DIR = 'datasets/'
SCALE_FACTOR = 5  # How much bigger the new dataset should be
SAVE_ORIGINAL = True
TARGET_DIMENSIONS = (350, 60)

with open(IN_JSON, 'r') as json_file:
    json_images = json.load(json_file)

image_paths = [os.path.join(IMG_IN_DIR, os.path.basename(i['path'])) for i in json_images]

seq = iaa.Sequential([
    # Small blur on half of the images
    iaa.Sometimes(0.5,
                  iaa.GaussianBlur(sigma=(0, 0.5))
                  ),
    iaa.ContrastNormalization((0.75, 1.5)),  # Change in contrast
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # Gaussian Noise can change colour
    iaa.Multiply((0.8, 1.2), per_channel=0.2),  # brighter/darker
    iaa.Affine(
        rotate=(-10, 10),
        shear=(-8, 8),
        fit_output=True
    ),  # TODO add piecewise affine
], random_order=True)  # apply augmenters in random order

images = np.array([np.array(Image.open(img)) for img in image_paths[:3]])  # TODO: remove range
images = np.repeat(images, SCALE_FACTOR, axis=0)
image_paths = np.repeat(np.array(image_paths), SCALE_FACTOR, axis=0)

# Make batches out of single images because processing multiple images at once leads to errors because the resulting
# images have different shapes and therefore can't be organised in a numpy array -> leads to an internal error
for idx, img in enumerate(images):
    img_aug = seq(images=np.expand_dims(img, axis=0))
    pil_img = Image.fromarray(np.squeeze(img_aug), 'L')
    # pil_img.show()
    padded_img = resize_img(pil_img, TARGET_DIMENSIONS)
    # padded_img.show()
    old_filename = os.path.splitext(os.path.basename(image_paths[idx]))[0]
    new_filename = old_filename + '-' + str(idx % SCALE_FACTOR + 1) + '.png'
    new_img_path = os.path.join(IMG_OUT_DIR, new_filename)
    # print(new_img_path)
    with open(new_img_path, 'wb+') as img_file:
        padded_img.save(img_file, format='PNG')

