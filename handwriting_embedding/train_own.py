import argparse
import configparser
import json
from math import sqrt

import cv2
import matplotlib
import numpy as np
import os
import random
from PIL import Image
from chainer import Chain, training, report, cuda, backend, serializers, optimizers
from chainer import functions as F
from chainer.links.model.vision.resnet import _global_average_pooling_2d
from chainer.training import extensions
from tensorboardX import SummaryWriter

from handwriting_embedding import triplet
from handwriting_embedding.cluster_plotter import ClusterPlotter, draw_embeddings_cluster_with_images
from handwriting_embedding.eval_utils import get_embeddings
from handwriting_embedding.resnet import ResNet
from handwriting_embedding.triplet_iterator import TripletIterator

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


def generate_datasets(json_paths, dataset_dir):
    datasets = []
    for dataset_description_path in json_paths:
        with open(os.path.join(dataset_dir, dataset_description_path), 'r') as f:
            json_file = json.load(f)
        dataset = [(image_to_array(os.path.join(dataset_dir, sample['path'])), sample['type']) for sample in
                   json_file]  # TODO refactor into dict
        random.shuffle(dataset)
        datasets.append(dataset)
    classes = set([sample[1] for ds in datasets for sample in ds])

    train = []
    test = []
    min_length = min([len(ds) for ds in datasets])
    threshold = int(min_length * 0.9)
    for ds in datasets:
        train.extend(ds[:threshold])
        test.extend(ds[threshold:min_length])

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Config file for Training params such as epochs, batch size, lr, etc.")
    parser.add_argument("model_suffix", type=str, help="Suffix that should be added to the end of the model filename")
    parser.add_argument("dataset_dir", type=str,
                        help="Directory where the images and the dataset description is stored")
    parser.add_argument("json_files", nargs="+", type=str,
                        help="JSON files that conatin the string-path-type mapping for each sample")
    parser.add_argument("-r", "--retrain", action="store_true", help="Model will be trained from scratch")
    parser.add_argument("-md", "--model-dir", type=str, default="models",
                        help="Dir where models will be saved/loaded from")
    parser.add_argument("-rs", "--resnet-size", type=int, default="18", help="Size of the used ResNet model")
    parser.add_argument("-ld", "--log-dir", type=str, help="name of tensorboard logdir")
    args = parser.parse_args()

    # TODO: inversion of image colors?

    ###################### CONFIG ############################
    retrain = args.retrain
    resnet_size = args.resnet_size
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_name = 'iamondb_res' + str(resnet_size) + "_" + args.model_suffix
    base_model = PooledResNet(resnet_size)
    plot_loss = True

    json_files = args.json_files

    config = configparser.ConfigParser()
    config.read(args.config)

    batch_size = int(config['TRAINING']['batch_size'])
    epochs = int(config['TRAINING']['epochs'])
    lr = float(config['TRAINING']['lr'])
    # lr_interval = int(config['TRAINING']['lr_interval'])
    gpu = config['TRAINING']['gpu']

    xp = cuda.cupy if int(gpu) >= 0 else np

    if args.log_dir is not None:
        writer = SummaryWriter(os.path.join("runs/", args.log_dir))
    else:
        writer = SummaryWriter()

    print("RETRAIN:", str(retrain), "MODEL_NAME:", model_name, "BATCH_SIZE:", str(batch_size), "EPOCHS:", str(epochs))

    #################### DATASETS ###########################

    train, test, classes = generate_datasets(json_files, args.dataset_dir)
    print("Datasets loaded. train samples: {}, test samples: {}".format(len(train), len(test)))

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

        model = Classifier(base_model)

        optimizer = optimizers.Adam(alpha=lr)
        optimizer.setup(model)
        updater = triplet.Updater(train_iter, optimizer, device=gpu)

        evaluator = triplet.Evaluator(test_iter, model, device=gpu)

        trainer = get_trainer(updater, evaluator, epochs)
        if plot_loss:
            trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(ClusterPlotter(base_model, test_labels, test_triplet, batch_size, xp), trigger=(1, 'epoch'))

        trainer.run()

        serializers.save_npz(os.path.join(model_dir, model_name + '.npz'), base_model)
    else:
        serializers.load_npz(os.path.join(model_dir, model_name + '.npz'), base_model)

        if int(gpu) >= 0:
            backend.get_device(gpu).use()
            base_model.to_gpu()

        embeddings = get_embeddings(base_model, test_triplet, batch_size, xp)
        draw_embeddings_cluster_with_images('cluster_final.png', embeddings, test_labels, test_triplet,
                                            draw_images=False)
        draw_embeddings_cluster_with_images('cluster_final_with_images.png', embeddings, test_labels, test_triplet,
                                            draw_images=True)

        # Add embeddings to projector - TODO: refactor into method
        height_pad = 0
        width_pad = 0
        if test_triplet.shape[2] > test_triplet.shape[3]:
            width_pad = test_triplet.shape[2] - test_triplet.shape[3]
        else:
            height_pad = test_triplet.shape[3] - test_triplet.shape[2]

        square_imgs = np.pad(test_triplet, ((0, 0), (0, 0), (0, height_pad), (0, width_pad)), mode="constant")

        n = square_imgs.shape[0]
        max_dim = int(8192 // sqrt(n))
        resized_imgs = []
        for img in square_imgs:
            img = np.transpose(img, (2, 1, 0))
            resized_img = cv2.resize(img, dsize=(max_dim, max_dim), interpolation=cv2.INTER_CUBIC)
            resized_img = np.transpose(resized_img, (2, 1, 0))
            resized_imgs.append(resized_img)
        resized_imgs = np.asarray(resized_imgs)

        writer.add_embedding(embeddings, metadata=test_labels, label_img=resized_imgs)

    print("Done")


if __name__ == '__main__':
    main()
