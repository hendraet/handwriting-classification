import argparse
import csv
from datetime import datetime
import math

import copy
import os
import random

import numpy as np

from PIL import ImageDraw, Image


def normalise_dataset(dataset):
    for i, stroke_set in enumerate(dataset):
        stroke_coords = stroke_set[:, 1:]

        # relative to absolute
        absolute_stroke_coords = np.cumsum(stroke_coords, axis=0)

        # normalise, so every stroke coordinate is >= 0
        absolute_stroke_coords[:, 0] -= absolute_stroke_coords[:, 0].min()
        absolute_stroke_coords[:, 1] -= absolute_stroke_coords[:, 1].min()

        dataset[i][:, 1] = absolute_stroke_coords[:, 0]
        dataset[i][:, 2] = absolute_stroke_coords[:, 1]

    return dataset


def get_datastet_and_labels(args, dataset_prefix):
    dataset_path = os.path.join(args.dataset_dir, dataset_prefix + ".npy")
    dataset = np.load(dataset_path, allow_pickle=True)
    dataset = normalise_dataset(dataset)

    labels_path = os.path.join(args.description_dir, dataset_prefix + ".csv")
    with open(labels_path, 'r') as label_file:
        reader = csv.reader(label_file)
        num_labels = [row[0] for row in reader]

    return dataset, num_labels


def generate_date():
    # TODO: '%m/%d/%y', '%m/%d/%Y','%d. %B %Y', '%d. %b %Y', '%B %y', '%d. %B', '%d. %b'
    date_formats = ['%d.%m.%y', '%d.%m.%Y', '%d-%m-%y']

    start = datetime.strptime('01.01.1000', '%d.%m.%Y')
    end = datetime.strptime('01.01.2020', '%d.%m.%Y')
    delta = end - start

    rand_date = start + delta * random.random()

    return rand_date.strftime(random.choice(date_formats))


def create_image_from_strokes(orig_x, orig_y, stroke_ends):
    img = Image.new("RGB", (1000, 200), color=(255, 255, 255))  # TODO width height
    img_canvas = ImageDraw.Draw(img)
    resize_factor = 3
    x = copy.deepcopy(orig_x)
    y = copy.deepcopy(orig_y)

    y *= -1  # y-axis is inverted

    x += 10
    y += 40

    for i, point in enumerate(stroke_ends[:-1]):
        if stroke_ends[i] == 1:  # TODO: check if correct
            continue
        img_canvas.line((x[i] * resize_factor, y[i] * resize_factor,
                         x[i + 1] * resize_factor, y[i + 1] * resize_factor),
                        fill=(0, 0, 0), width=3)
    img.show()


def concatenate_strokes(datasets, sequence_idx, show_image=False):
    # Copy is necessary because otherwise the padding is messed up if the same stroke_set is used twice
    stroke_sets = np.asarray([copy.deepcopy(datasets[i]) for i in sequence_idx])

    absolute_strokes = []
    for stroke_set in stroke_sets:
        stroke_coords = stroke_set[:, 1:]

        # shift current stroke set, so that it is displayed on the right of the previous stroke set
        if absolute_strokes:
            padding = 5
            stroke_coords[:, 0] += absolute_strokes[-1][:, 0].max() + padding

        absolute_strokes.append(stroke_coords)

    stroke_ends = np.concatenate([el[:, 0] for el in stroke_sets])
    absolute_strokes = np.concatenate(absolute_strokes)

    x = absolute_strokes[:, 0]
    y = absolute_strokes[:, 1]

    if show_image:
        create_image_from_strokes(x, y, stroke_ends)

    return x, y, stroke_ends


def get_indices_for_string(num_labels, string):
    indices = []

    for char in string:
        matching_indices = [i for i, el in enumerate(num_labels) if el == char]
        selected_idx = random.choice(matching_indices)
        indices.append(selected_idx)

    return indices


def write_results(samples, labels):
    np.save("dates.npy", np.asarray(samples))
    with open("dates.csv", "w") as label_file:
        label_file.write("\n".join(labels) + "\n")


def convert_from_absoulte_to_relative(x, y):
    combined = np.stack([x, y], axis=1)
    combined_relative = combined[1:] - combined[:-1]
    combined_relative = np.insert(combined_relative, 0, [0., 0.], axis=0)

    return combined_relative[:, 0], combined_relative[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="../datasets")
    parser.add_argument("--description-dir", default="../dataset_descriptions")
    parser.add_argument("--num-dataset", default="iamondb_num")
    parser.add_argument("--dot-dataset", default="iamondb_dot")
    parser.add_argument("--dash-dataset", default="iamondb_dash")
    parser.add_argument("--num", type=int, default="10")
    parser.add_argument("--show-image", action="store_true")
    args = parser.parse_args()

    # tmp = np.load("../../../Handwriting-synthesis/data/strokes.npy", allow_pickle=True, encoding="bytes")
    # tmpy = np.load("../datasets/iamondb_num.npy", allow_pickle=True, encoding="bytes")
    # tmp = normalise_dataset(tmp)
    # tmpy = normalise_dataset(tmpy)
    # create_image_from_strokes(tmp[0][:,1],tmp[0][:,2],tmp[0][:,0],)
    # create_image_from_strokes(tmpy[0][:,1],tmpy[0][:,2],tmpy[0][:,0],)

    num_dataset, num_labels = get_datastet_and_labels(args, args.num_dataset)
    dot_dataset, dot_labels = get_datastet_and_labels(args, args.dot_dataset)
    dash_dataset, dash_labels = get_datastet_and_labels(args, args.dash_dataset)

    # height = np.median([stroke_set[:,2].max() for stroke_set in num_dataset])  # TODO: better heuristic
    for i, stroke_set in enumerate(dash_dataset):
        stroke_set[:, 2] = stroke_set[:, 2] + 5
        dash_dataset[i] = stroke_set

    datasets = np.concatenate([num_dataset, dot_dataset, dash_dataset])
    labels = num_labels + dot_labels + dash_labels

    samples = []
    new_labels = []
    for i in range(0, args.num):
        date = generate_date()

        sequence_idx = get_indices_for_string(labels, date)

        x, y, stroke_ends = concatenate_strokes(datasets, sequence_idx, show_image=args.show_image)
        x, y = convert_from_absoulte_to_relative(x, y)

        new_stroke_set = np.stack((stroke_ends, x, y), axis=1)
        samples.append(new_stroke_set)
        new_labels.append(date)

    write_results(samples, new_labels)


if __name__ == '__main__':
    main()
