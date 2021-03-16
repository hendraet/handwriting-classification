import json
import os
import random
from PIL import Image
import numpy as np


def image_to_array(img, invert_colours=False):
    img_array = np.array(img, dtype='float32')
    img_array /= 255
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    if invert_colours:
        img_array = 1 - img_array
    return np.transpose(img_array, (2, 0, 1))


def image_array_from_path(path, invert_colours=False):
    img = Image.open(path)
    return image_to_array(img, invert_colours)


def parse_json(dataset_dir, json_path):
    classes = set()
    dataset = []

    with open(os.path.join(dataset_dir, json_path), "r") as json_file:
        json_content = json.load(json_file)
        num_samples = len(json_content)
        for i, sample in enumerate(json_content):
            if (i + 1) % 250 == 0:
                print(f"Loading image {i + 1} of {num_samples}")
            img = image_array_from_path(os.path.join(dataset_dir, sample['path']), invert_colours=True)
            dataset.append((img, sample["type"]))
            classes.add(sample["type"])

    return dataset, classes


def generate_positives_negatives(anchors):
    positives = []
    negatives = []
    for sample in anchors:
        sample_class = sample[1]
        positive_found = False
        negative_found = False
        # TODO: fallback for dataset that is too small
        while not (positive_found and negative_found):
            sample_idx = random.randint(0, len(anchors) - 1)
            candidate = anchors[sample_idx]
            if not positive_found and candidate[1] == sample_class and not np.array_equal(sample[0], candidate[0]):
                positives.append(candidate)
                positive_found = True
            if not negative_found and candidate[1] != sample_class:
                negatives.append(candidate)
                negative_found = True

    assert len(positives) == len(negatives) and len(anchors) == len(negatives)
    return positives, negatives


def generate_triplet(dataset):
    anchors = dataset
    positives, negatives = generate_positives_negatives(anchors)

    datasets = [anchors, positives, negatives]
    merged = [None] * 3 * len(anchors)
    for i in range(0, 3):
        merged[i::3] = datasets[i]

    triplet, labels = zip(*merged)
    assert not [i for (i, label) in enumerate(labels[0::3]) if label == labels[i * 3 + 2]]
    return np.asarray(triplet)


def load_dataset(args):
    train, classes = parse_json(args.dataset_dir, args.train_path)
    test, _ = parse_json(args.dataset_dir, args.test_path)
    print("Datasets loaded. train samples: {}, test samples: {}".format(len(train), len(test)))

    return train, test, sorted(classes)


def load_triplet_dataset(args):
    train, test, classes = load_dataset(args)

    train_triplet = generate_triplet(train)
    train_samples, train_labels = zip(*train)
    print("Train done.")

    test_triplet = generate_triplet(test)
    test_samples, test_labels = zip(*test)
    print("Test done.")

    return train_triplet, np.asarray(train_samples), np.asarray(train_labels), \
           test_triplet, np.asarray(test_samples), np.asarray(test_labels)
