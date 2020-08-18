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
    anchors = generate_triplet_part(dataset, False, classes)
    positives = generate_triplet_part(dataset, False, classes)
    negatives = generate_triplet_part(dataset, True, classes)

    datasets = [anchors, positives, negatives]
    merged = [None] * 3 * len(anchors)
    for i in range(0, 3):
        merged[i::3] = datasets[i]

    triplet, labels = zip(*merged)
    return np.asarray(triplet), np.asarray(labels)


def load_dataset(args):
    train, classes = parse_json(args.dataset_dir, args.train_path)
    test, _ = parse_json(args.dataset_dir, args.test_path)
    print("Datasets loaded. train samples: {}, test samples: {}".format(len(train), len(test)))

    train_triplet, train_labels = generate_triplet(train, classes)
    assert not [i for (i, label) in enumerate(train_labels[0::3]) if label == train_labels[i * 3 + 2]]
    print("Train done.")

    test_triplet, test_labels = generate_triplet(test, classes)
    assert not [i for (i, label) in enumerate(test_labels[0::3]) if label == test_labels[i * 3 + 2]]
    print("Test done.")

    return train_triplet, train_labels, test_triplet, test_labels
