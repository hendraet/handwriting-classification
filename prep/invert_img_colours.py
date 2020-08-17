import os
from PIL import Image, ImageOps

dataset_dir = "/home/hendrik/experiments/handwriting_embedding/datasets"

for filename in os.listdir(dataset_dir):
    full_filename = os.path.join(dataset_dir, filename)
    if os.path.splitext(filename)[1] != ".png" or os.path.isdir(full_filename):
        print(f"Skipping {filename}")
        continue

    img = Image.open(full_filename)
    inverted_image = ImageOps.invert(img)
    inverted_image.save(full_filename)

