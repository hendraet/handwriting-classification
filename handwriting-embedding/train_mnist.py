"""This script serves as a simple example and test for the DNN"""
import configparser
import itertools
import random
import sys

import numpy as np
import triplet
from chainer import Chain
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import optimizers, report, training
from chainer.datasets import get_mnist
from chainer.training import extensions
from triplet_iterator import TripletIterator


class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y


class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x_a, x_p, x_n):
        y_a, y_p, y_n = (self.predictor(x) for x in (x_a, x_p, x_n))
        loss = F.triplet(y_a, y_p, y_n)
        report({'loss': loss}, self)
        return loss


def usage():
    print("python3 {} <config_file.ini>".format(sys.argv[0]))


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


def order_set(dataset):
    dataset_sorted = sorted(dataset, key=lambda tup: tup[1])

    dataset_one = [dataset_sorted[i] for i in range(0, len(dataset_sorted), 3)]
    dataset_two = [dataset_sorted[i] for i in range(1, len(dataset_sorted), 3)]
    dataset_three = [dataset_sorted[i] for i in range(2, len(dataset_sorted), 3)]

    # random.shuffle(dataset_three)
    dataset_shuffled = []
    block_size = int(len(dataset_three) / 10)  # Only works well enough if divisble by 10 obviously
    for i in range(0, 10):
        first = dataset_three[:i * block_size]
        second = dataset_three[(i + 1) * block_size:]
        combined = first + second
        sample = random.sample(combined, block_size)
        dataset_shuffled.extend(sample)

    iters = [iter(dataset_one), iter(dataset_two), iter(dataset_shuffled)]
    merged = list(next(it) for it in itertools.cycle(iters))

    return [tup[0] for tup in merged]


def main():
    if not len(sys.argv) == 2:
        usage()
        exit(1)

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    batch_size = int(config['TRAINING']['batch_size'])
    epochs = int(config['TRAINING']['epochs'])
    lr = float(config['TRAINING']['lr'])
    lr_interval = int(config['TRAINING']['lr_interval'])
    gpu = int(config['TRAINING']['gpu'])

    xp = cuda.cupy if gpu >= 0 else np

    # TODO: change GPU in conf
    train, test = get_mnist(withlabel=True)

    train_merged = order_set(train)

    # TODO: Doesn't work out too well for test set (some points are missing)
    # test = test[:-1]
    test_merged = order_set(test)
    test_merged = test_merged[:-2]

    train_iter = TripletIterator(train_merged,
                                 batch_size=batch_size,
                                 repeat=True,
                                 xp=xp)
    test_iter = TripletIterator(test_merged,
                                batch_size=batch_size,
                                xp=xp)
    base_model = MLP(100, 10)
    model = Classifier(base_model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    optimizer = optimizers.SGD(lr=lr)
    optimizer.setup(model)
    updater = triplet.Updater(train_iter, optimizer)

    evaluator = triplet.Evaluator(test_iter, model)

    trainer = get_trainer(updater, evaluator, epochs)
    trainer.run()

    # for img in test_merged:
    #     embedding = model(img)


if __name__ == '__main__':
    main()
