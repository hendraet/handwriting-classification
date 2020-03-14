import argparse
import csv

import os

import numpy as np

# generate random date string (without dashes first)
# get random stroke for each char
# Normalise height (and width)?
# glue them together:
#   get rectangle bounding box for each char
#       for dots: add a little bit of padding to the bottom and lot to the top in total as high as highest (?) num (double check if dots even start at the bottom)
#       for dashes: equal padding to top and bottom in total as high as highest
#       rest: straigh forward
#   padding to the right between individual chars
# plot (later save as strokes)
from PIL import ImageDraw, Image


def create_image_from_strokes(x, y, stroke_ends):
    img = Image.new("RGB", (500, 500), color=(255, 255, 255))  # TODO width height
    img_canvas = ImageDraw.Draw(img)
    resize_factor = 1

    y *= -1  # y-axis is inverted

    x += 100
    y += 100

    for i, point in enumerate(stroke_ends[:-1]):
        if stroke_ends[i] == 1:  # TODO: check if correct
            continue
        img_canvas.line((x[i] * resize_factor, y[i] * resize_factor,
                         x[i + 1] * resize_factor, y[i + 1] * resize_factor),
                        fill=(0, 0, 0), width=3)
    img.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dad", "--dataset-dir", default="../datasets")
    parser.add_argument("-ded", "--description-dir", default="../dataset_descriptions")
    parser.add_argument("-nd", "--num-dataset", default="iamondb_num")
    args = parser.parse_args()

    num_dataset_path = os.path.join(args.dataset_dir, args.num_dataset + ".npy")
    num_dataset = np.load(num_dataset_path, allow_pickle=True)

    num_labels_path = os.path.join(args.description_dir, args.num_dataset + ".csv")
    with open(num_labels_path, 'r') as num_label_file:
        reader = csv.reader(num_label_file)
        num_labels = [row[0] for row in reader]

    sequence_idx = [6, 0, 1, 25]

    stroke_sets = np.asarray([num_dataset[i] for i in sequence_idx])

    # relative to absolute
    absolute_strokes = []
    for stroke_set in stroke_sets:
        stroke_coords = stroke_set[:, 1:]
        absolute_stroke_coords = np.cumsum(stroke_coords, axis=0)

        # normalise, so every stroke coordinate is >= 0
        absolute_stroke_coords[:,0] -= absolute_stroke_coords[:,0].min()
        absolute_stroke_coords[:,1] -= absolute_stroke_coords[:,1].min()

        # shift x coords of stroke_set by max value of previous stroke set, so chars are display next to each other
        if absolute_strokes:
            absolute_stroke_coords[:, 0] += absolute_strokes[-1][:,0].max() + 50

        absolute_strokes.append(absolute_stroke_coords)

    stroke_ends = np.concatenate([el[:,0] for el in stroke_sets])

    absolute_strokes = np.concatenate(absolute_strokes)
    x = absolute_strokes[:, 0]
    y = absolute_strokes[:, 1]

    # stacked_strokes = np.concatenate(strokes)
    # stroke_ends = stacked_strokes[:,0]
    # stroke_coords = stacked_strokes[:,1:]
    # absolute_stroke_coords = np.cumsum(stroke_coords, axis=0)

    create_image_from_strokes(x, y, stroke_ends)

    # get bounding rect
    # sample = num_dataset[0]
    # x = np.cumsum(sample[:, 1])
    # y = np.cumsum(sample[:, 2])
    # bounding_rect = (x.min(), y.min(), x.max(), y.max())
    pass


if __name__ == '__main__':
    main()
