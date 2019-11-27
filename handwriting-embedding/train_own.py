import configparser
import itertools
import json
import random

import numpy as np
import triplet

from PIL import Image
from chainer import Chain, training, report, cuda, backend, serializers, optimizers
from chainer.training import extensions
from chainer import functions as F
from triplet_iterator import TripletIterator
from resnet import ResNet


class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x_a, x_p, x_n):
        y_a, y_p, y_n = (self.predictor(x) for x in (x_a, x_p, x_n))
        loss = F.triplet(y_a, y_p, y_n)
        report({'loss': loss}, self)
        return loss


def get_trainer(updater, evaluator, epochs):
    trainer = training.Trainer(updater, (epochs, 'epoch'), out='result')
    trainer.extend(evaluator)
    # TODO: reduce LR -- how to update every X epochs?
    # trainer.extend(extensions.ExponentialShift('lr', 0.1, target=lr*0.0001))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(
        (epochs, 'epoch'), update_interval=10))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    return trainer


def image_to_array(path):
    img = Image.open(path)
    img_array = np.array(img, dtype='float32')
    img_array /= 255
    return np.transpose(img_array, (2, 0, 1))


def generate_datasets(json_paths):
    train = []
    test = []

    for dataset_path in json_paths:
        with open(dataset_path, 'r') as f:
            json_file = json.load(f)
            dataset = [(image_to_array(sample['path']), 'num' if sample['type'] == 'num' else 'text') for sample in
                       json_file]
            threshold = int(len(dataset) * 0.9)
            # Datasets should be shuffled if generated differently
            train.extend(dataset[:threshold])
            test.extend(dataset[threshold:])

    random.shuffle(train)
    random.shuffle(test)

    return train, test


def generate_triplet_part(dataset, negative, classes):
    triplet_part = []
    num_classes = len(classes)
    for cl in classes:
        for j in range(0, int(len(dataset) / (3 * num_classes))):
            while True:
                sample_idx = random.randint(0, len(dataset) - 1)
                sample = dataset[sample_idx]
                if negative:
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

    iters = [iter(anchors), iter(positives), iter(negatives)]
    merged = list(next(it) for it in itertools.cycle(iters))

    return zip(*merged)


def main():
    ###################### CONFIG ############################

    config = configparser.ConfigParser()
    config.read('mnist.conf')

    batch_size = int(config['TRAINING']['batch_size'])
    epochs = int(config['TRAINING']['epochs'])
    lr = float(config['TRAINING']['lr'])
    lr_interval = int(config['TRAINING']['lr_interval'])
    gpu = int(config['TRAINING']['gpu'])

    xp = cuda.cupy if gpu >= 0 else np

    #################### DATASETS ###########################

    classes = ['num', 'text']
    train, test = generate_datasets(['datasets/num.json', 'datasets/texts.json'])

    train_triplet, train_labels = generate_triplet(train, classes)
    assert not [i for (i, label) in enumerate(train_labels[0::3]) if label == train_labels[i * 3 + 2]]
    print("Train Done")

    test_triplet, test_labels = generate_triplet(test, classes)
    assert not [i for (i, label) in enumerate(test_labels[0::3]) if label == test_labels[i * 3 + 2]]
    print("Test Done")

    #################### Train and Save Model ########################################

    train_iter = TripletIterator(train_triplet,
                                 batch_size=batch_size,
                                 repeat=True,
                                 xp=xp)
    test_iter = TripletIterator(test_triplet,
                                batch_size=batch_size,
                                xp=xp)
    base_model = ResNet(18)
    model = Classifier(base_model)

    if gpu >= 0:
        backend.get_device(gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam(alpha=lr)
    optimizer.setup(model)
    updater = triplet.Updater(train_iter, optimizer)

    evaluator = triplet.Evaluator(test_iter, model)

    trainer = get_trainer(updater, evaluator, epochs)
    trainer.run()

    # serializers.save_npz('full_model.npz', model)
    serializers.save_npz('embeddings_resnet_own.npz', base_model)


if __name__ == '__main__':
    main()
