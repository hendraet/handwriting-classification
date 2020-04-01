import argparse
import csv
from datetime import datetime
import math

import copy
import json
import os
import random

import numpy as np
import statistics

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


def create_image_from_strokes(orig_x, orig_y, stroke_ends, show_image):
    x = copy.deepcopy(orig_x)
    y = copy.deepcopy(orig_y)

    padding = 10
    resize_factor = 3

    y *= -1  # y-axis is inverted
    y -= y.min()

    x *= resize_factor
    y *= resize_factor

    x += padding
    y += padding

    width = math.ceil(x.max() + padding)
    height = math.ceil(y.max() + padding)

    img = Image.new("RGB", (width, height), color=(255, 255, 255))  # TODO width height
    img_canvas = ImageDraw.Draw(img)

    for i, point in enumerate(stroke_ends[:-1]):
        if stroke_ends[i] == 1:  # TODO: check if correct
            continue
        img_canvas.line((x[i], y[i], x[i + 1], y[i + 1]), fill=(0, 0, 0), width=3)

    if show_image:
        img.show()

    return img


def concatenate_strokes(string, datasets, sequence_idx, resize=True):
    # Copy is necessary because otherwise the padding is messed up if the same stroke_set is used twice
    stroke_sets = np.asarray([copy.deepcopy(datasets[i]) for i in sequence_idx])

    heights = [(stroke_set[:, 2].max() - stroke_set[:, 2].min()) for stroke_set in stroke_sets]
    # Exclude punctuation form median calculation to avoid unnecessary distortion
    median_height = statistics.mean([height for i, height in enumerate(heights) if string[i].isdigit()])

    absolute_strokes = []
    for i, stroke_set in enumerate(stroke_sets):
        stroke_coords = stroke_set[:, 1:]

        if string[i].isdigit() and resize:
            distortion_factor = random.uniform(-0.10, 0.10)
            resize_factor = median_height / heights[i] + distortion_factor
            stroke_coords *= resize_factor

        # shift current stroke set, so that it is displayed on the right of the previous stroke set
        if absolute_strokes:
            padding = 5
            stroke_coords[:, 0] += absolute_strokes[-1][:, 0].max() + padding

        absolute_strokes.append(stroke_coords)

    stroke_ends = np.concatenate([el[:, 0] for el in stroke_sets])
    absolute_strokes = np.concatenate(absolute_strokes)

    x = absolute_strokes[:, 0]
    y = absolute_strokes[:, 1]

    return x, y, stroke_ends


def get_indices_for_string(num_labels, string):
    indices = []

    for char in string:
        matching_indices = [i for i, el in enumerate(num_labels) if el == char]
        selected_idx = random.choice(matching_indices)
        indices.append(selected_idx)

    return indices


def write_synth_results(samples, labels, synth_ready):
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
    parser.add_argument("--synth-ready", action="store_true",
                        help="Csv and npy files will be generated JSON and images otherwise")
    args = parser.parse_args()

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
    dataset_info = []
    for i in range(0, args.num):
        string = generate_date()

        sequence_idx = get_indices_for_string(labels, string)

        x, y, stroke_ends = concatenate_strokes(string, datasets, sequence_idx, resize=True)
        img = create_image_from_strokes(x, y, stroke_ends, args.show_image)
        x, y = convert_from_absoulte_to_relative(x, y)

        if args.synth_ready:
            new_stroke_set = np.stack((stroke_ends, x, y), axis=1)
            samples.append(new_stroke_set)
            dataset_info.append(string)
        else:
            filename = string + "_" + "_".join([str(i) for i in (sequence_idx)]) + ".png"
            out_path = os.path.join(args.dataset_dir, filename)
            with open(out_path, "wb") as out_file:
                img.save(out_file)

            info = {
                "string": string,
                "type": "date",
                "path": out_path.replace("../", ""),
            }
            dataset_info.append(info)


    if args.synth_ready:
        write_synth_results(samples, dataset_info, args.synth_ready)
    else:
        with open(os.path.join(args.description_dir, "iamondb_generated_dates.json"), "w") as out_json:
            json.dump(dataset_info, out_json, indent=4)


if __name__ == '__main__':
    main()
