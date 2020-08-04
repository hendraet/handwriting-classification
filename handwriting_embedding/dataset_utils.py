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


def generate_datasets(json_paths, dataset_dir):
    dataset = {}
    for dataset_description_path in json_paths:
        with open(os.path.join(dataset_dir, dataset_description_path), 'r') as f:
            json_file = json.load(f)

        for sample in json_file:
            # colour inversion, so that the background has a value of 0 and represent "no information"
            img = image_to_array(os.path.join(dataset_dir, sample['path']), invert_colours=True)
            if sample['type'] in dataset:
                dataset[sample['type']].append(img)
            else:
                dataset[sample['type']] = [img]
    classes = dataset.keys()

    train = []
    test = []
    smallest_class_len = min([len(ds) for ds in dataset.values()])
    threshold = int(smallest_class_len * 0.9)
    for class_name, imgs in dataset.items():
        random.shuffle(imgs)  # Makes sure that no internal structure in the json file messes up dataset
        train.extend([(img, class_name) for img in imgs[:threshold]])
        test.extend([(img, class_name) for img in imgs[threshold:smallest_class_len]])

    random.shuffle(train)
    random.shuffle(test)

    return train, test, classes


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


def load_dataset(json_files, args):
    train, test, classes = generate_datasets(json_files, args.dataset_dir)
    print("Datasets loaded. train samples: {}, test samples: {}".format(len(train), len(test)))

    # mnist_dataset = get_mnist(ndim=3)
    # train = [(np.tile(sample[0], (3, 1, 1)), str(sample[1])) for sample in mnist_dataset[0]]
    # test = [(np.tile(sample[0], (3, 1, 1)), str(sample[1])) for sample in mnist_dataset[1]]
    # classes = set([label for img, label in train])

    train_triplet, train_labels = generate_triplet(train, classes)
    assert not [i for (i, label) in enumerate(train_labels[0::3]) if label == train_labels[i * 3 + 2]]
    print("Train done.")

    test_triplet, test_labels = generate_triplet(test, classes)
    assert not [i for (i, label) in enumerate(test_labels[0::3]) if label == test_labels[i * 3 + 2]]
    print("Test done.")

    return train_triplet, train_labels, test_triplet, test_labels
