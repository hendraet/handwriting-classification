import json
import os
import random
from PIL import Image
import numpy as np


def image_to_array(path, invert_colours=False):
    img = Image.open(path)
    img_array = np.array(img, dtype='float32')
    img_array /= 255
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    if invert_colours:
        img_array = 1 - img_array
    return np.transpose(img_array, (2, 0, 1))


def parse_json(dataset_dir, json_path):
    classes = set()
    dataset = []

    with open(os.path.join(dataset_dir, json_path), "r") as json_file:
        json_content = json.load(json_file)
        for sample in json_content:
            img = image_to_array(os.path.join(dataset_dir, sample['path']), invert_colours=True)
            dataset.append((img, sample["type"]))
            classes.add(sample["type"])

    return dataset, classes


def generate_triplet_part(dataset, is_negative, classes):
    triplet_part = []
    num_classes = len(classes)
    for cl in classes:
        for j in range(0, int(len(dataset) / (3 * num_classes))):
            while True:
                sample_idx = random.randint(0, len(dataset) - 1)
                sample = dataset[sample_idx]
                if is_negative:
                    if sample[1] != cl:
                        triplet_part.append(sample)
                        break
                else:
                    if sample[1] == cl:
                        triplet_part.append(sample)
                        break
    return triplet_part


def generate_triplet(dataset, classes):
    num_samples = len(dataset)
    ds_range = range(num_samples)

    indices = []
    for anchor_idx in ds_range:
        for positive_idx in ds_range:
            if anchor_idx == positive_idx or dataset[anchor_idx][1] != dataset[positive_idx][1]:
                continue
            for negative_idx in ds_range:
                if dataset[anchor_idx][1] == dataset[negative_idx][1]:
                    continue
                indices.extend([anchor_idx, positive_idx, negative_idx])

    return np.asarray(indices)


def load_dataset(args):
    train, classes = parse_json(args.dataset_dir, args.train_path)
    test, _ = parse_json(args.dataset_dir, args.test_path)
    print("Datasets loaded. train samples: {}, test samples: {}".format(len(train), len(test)))

    return train, test, classes


def load_triplet_dataset(args):
    train, test, classes = load_dataset(args)

    train_indices = generate_triplet(train, classes)
    print("Train done.")

    test_indices = generate_triplet(test, classes)
    print("Test done.")

    return train_indices, train, test_indices, test
