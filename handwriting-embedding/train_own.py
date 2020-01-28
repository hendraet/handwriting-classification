import configparser
import itertools
import json

import matplotlib
import numpy as np
import random
import triplet
from PIL import Image
from chainer import Chain, training, report, cuda, backend, serializers, optimizers
from chainer import functions as F
from chainer.links.model.vision.resnet import _global_average_pooling_2d
from chainer.training import extensions
from cluster_plotter import ClusterPlotter, draw_embeddings_cluster_with_images, draw_embeddings_cluster
from resnet import ResNet
from triplet_iterator import TripletIterator

matplotlib.use('Agg')


class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x_a, x_p, x_n):
        y_a, y_p, y_n = (self.predictor(x) for x in (x_a, x_p, x_n))
        loss = F.triplet(y_a, y_p, y_n, margin=1)
        report({'loss': loss}, self)
        return loss


class PooledResNet(Chain):
    def __init__(self, n_layers):
        super(PooledResNet, self).__init__()

        with self.init_scope():
            self.feature_extractor = ResNet(n_layers)

    def __call__(self, x):
        h = self.feature_extractor(x)
        h = _global_average_pooling_2d(h)
        return h


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
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    return np.transpose(img_array, (2, 0, 1))


def generate_datasets(json_paths):
    train = []
    test = []

    for dataset_path in json_paths:
        with open(dataset_path, 'r') as f:
            json_file = json.load(f)
            dataset = [(image_to_array(sample['path']), sample['type']) for sample in json_file]
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
    merged = list(next(it) for it in itertools.cycle(iters))  # Works until python 3.6

    return zip(*merged)


def main():
    ###################### CONFIG ############################

    retrain = False
    model_name = 'iamdb_nums_vs_txt'
    plot_loss = True

    json_files = ['datasets/iamdb_nums_aug.json', 'datasets/iamdb_words_aug.json']
    classes = ['num', 'text']

    config = configparser.ConfigParser()
    config.read('own.conf')

    batch_size = int(config['TRAINING']['batch_size'])
    epochs = int(config['TRAINING']['epochs'])
    lr = float(config['TRAINING']['lr'])
    lr_interval = int(config['TRAINING']['lr_interval'])
    gpu = config['TRAINING']['gpu']

    xp = cuda.cupy if int(gpu) >= 0 else np

    print("RETRAIN:", str(retrain), "MODEL_NAME:", model_name, "BATCH_SIZE:", str(batch_size), "EPOCHS:", str(epochs))

    #################### DATASETS ###########################

    train, test = generate_datasets(json_files)
    print("Datasets loaded.")

    train_triplet, train_labels = generate_triplet(train, classes)
    assert not [i for (i, label) in enumerate(train_labels[0::3]) if label == train_labels[i * 3 + 2]]
    print("Train done.")

    test_triplet, test_labels = generate_triplet(test, classes)
    assert not [i for (i, label) in enumerate(test_labels[0::3]) if label == test_labels[i * 3 + 2]]
    print("Test done.")

    #################### Train and Save Model ########################################

    if retrain:
        train_iter = TripletIterator(train_triplet,
                                     batch_size=batch_size,
                                     repeat=True,
                                     xp=xp)
        test_iter = TripletIterator(test_triplet,
                                    batch_size=batch_size,
                                    xp=xp)

        base_model = PooledResNet(18)
        model = Classifier(base_model)

        optimizer = optimizers.Adam(alpha=lr)
        optimizer.setup(model)
        updater = triplet.Updater(train_iter, optimizer, device=gpu)

        evaluator = triplet.Evaluator(test_iter, model, device=gpu)

        trainer = get_trainer(updater, evaluator, epochs)
        if plot_loss:
            trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(ClusterPlotter(base_model, test_labels, test_triplet, xp), trigger=(1, 'epoch'))

        trainer.run()

        serializers.save_npz(model_name + '.npz', base_model)
    else:
        base_model = PooledResNet(18)
        serializers.load_npz(model_name + '.npz', base_model)

        if int(gpu) >= 0:
            backend.get_device(gpu).use()
            base_model.to_gpu()
        draw_embeddings_cluster_with_images('cluster_final.png', base_model, test_labels, test_triplet, xp,
                                            draw_images=False)
        draw_embeddings_cluster_with_images('cluster_final_with_images.png', base_model, test_labels, test_triplet, xp,
                                            draw_images=True)

    print("Done")


if __name__ == '__main__':
    main()
