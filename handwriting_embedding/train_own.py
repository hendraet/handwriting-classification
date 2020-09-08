import argparse
import configparser
import glob
import json
import math
import sys

import matplotlib
import numpy as np
import os
import shutil
from chainer import training, cuda, backend, serializers, optimizers
from chainer.iterators import SerialIterator
from chainer.training import extensions, StandardUpdater, triggers
from tensorboardX import SummaryWriter

import triplet
from ce_evaluator import CEEvaluator
from classification import evaluate_embeddings, get_metrics
from dataset_utils import load_triplet_dataset, load_dataset
from eval_utils import create_tensorboard_embeddings
from eval_utils import get_embeddings
from extensions.cluster_plotter import ClusterPlotter
from models import PooledResNet
from models import StandardClassifier, LosslessClassifier, CrossEntropyClassifier
from triplet_iterator import TripletIterator

matplotlib.use("Agg")


def get_trainer(updater, evaluator, epochs):
    trainer = training.Trainer(updater, (epochs, "epoch"), out="result")
    trainer.extend(evaluator)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar((epochs, "epoch"), update_interval=10))
    trainer.extend(extensions.PrintReport(["epoch", "main/loss", "validation/main/loss"]))
    return trainer


def evaluate_triplet(model, test_triplet, test_labels, batch_size, writer, xp):
    embeddings = get_embeddings(model.predictor, test_triplet, batch_size, xp)
    # draw_embeddings_cluster_with_images("cluster_final.png", embeddings, test_labels, test_triplet,
    #                                     draw_images=False)
    # draw_embeddings_cluster_with_images("cluster_final_with_images.png", embeddings, test_labels, test_triplet,
    #                                     draw_images=True)

    # Add embeddings to projector
    test_triplet = 1 - test_triplet  # colours are inverted for model - "re-invert" for better visualisation
    create_tensorboard_embeddings(test_triplet, test_labels, embeddings, writer)

    metrics = evaluate_embeddings(embeddings, test_labels)
    return metrics


def evaluate_ce(model, test, batch_size, label_map, xp):
    test_samples, test_labels = zip(*test)

    predictions = []
    for i in range(math.ceil(len(test_samples) / batch_size)):
        predictions.append(model.predict(xp.asarray(list(test_samples[i * batch_size:(i + 1) * batch_size]))))
    predictions = xp.concatenate(predictions)
    if xp == cuda.cupy:
        predictions = xp.asnumpy(predictions)
    metrics = get_metrics(predictions, test_labels, list(label_map.keys()))

    inv_label_map = {v: k for k, v in label_map.items()}
    metrics["predicted_distribution"] = {inv_label_map[k]: v for k, v in metrics["predicted_distribution"].items()}
    return metrics


def main():
    # TODO: cleanup and move to conf
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Config file for Training params such as epochs, batch size, lr, etc.")
    parser.add_argument("model_suffix", type=str, help="Suffix that should be added to the end of the model filename")
    parser.add_argument("dataset_dir", type=str,
                        help="Directory where the images and the dataset description is stored")
    parser.add_argument("train_path", type=str, help="path to JSON file containing train set information")
    parser.add_argument("test_path", type=str, help="path to JSON file containing test set information")
    parser.add_argument("-rs", "--resnet-size", type=int, default="18", help="Size of the used ResNet model")
    parser.add_argument("-ld", "--log-dir", type=str, help="name of tensorboard logdir")
    parser.add_argument("-ll", "--lossless", action="store_true",
                        help="use lossless triplet loss instead of standard one")
    parser.add_argument("-ce", "--ce-classifier", action="store_true",
                        help="use a cross entropy classifier instead of triplet loss")
    parser.add_argument("-eo", "--eval-only", type=str, help="only evaluate the given model")
    args = parser.parse_args()

    ###################### INIT ############################
    resnet_size = args.resnet_size
    base_model = PooledResNet(resnet_size)

    # parse config file
    plot_loss = True
    config = configparser.ConfigParser()
    config.read(args.config)

    batch_size = int(config["TRAINING"]["batch_size"])
    epochs = int(config["TRAINING"]["epochs"])
    lr = float(config["TRAINING"]["lr"])
    # lr_interval = int(config["TRAINING"]["lr_interval"])
    gpu = config["TRAINING"]["gpu"]

    xp = cuda.cupy if int(gpu) >= 0 else np

    model_name = f"res{str(resnet_size)}_{args.model_suffix}"

    # Init tensorboard writer
    if args.log_dir is not None:
        if args.eval_only is not None:
            log_dir = f"runs/{args.log_dir}_eval"
        else:
            log_dir = f"runs/{args.log_dir}_ep{epochs}"
        if os.path.exists(log_dir):
            user_input = input("Log dir not empty. Clear log dir? (y/N)")
            if user_input == "y":
                shutil.rmtree(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = SummaryWriter()

    with open(os.path.join(writer.logdir, "args.log"), "w") as log_file:
        log_file.write(f"{' '.join(sys.argv[1:])}\n")
    shutil.copy(args.config, writer.logdir)

    print("MODEL_NAME:", model_name, "BATCH_SIZE:", str(batch_size), "EPOCHS:", str(epochs))

    #################### Train and Save Model ########################################
    if args.ce_classifier:
        train, test, classes = load_dataset(args)

        # convert labels from string to int
        label_map = {label: i for i, label in enumerate(classes)}
        train = [(sample, label_map[label]) for sample, label in train]
        test = [(sample, label_map[label]) for sample, label in test]

        train_iter = SerialIterator(train, batch_size, repeat=True, shuffle=True)
        test_iter = SerialIterator(test, batch_size, repeat=False, shuffle=True)

        model = CrossEntropyClassifier(base_model, len(classes))

        if int(gpu) >= 0:
            backend.get_device(gpu).use()
            base_model.to_gpu()
            model.to_gpu()

        optimizer = optimizers.Adam(alpha=lr)
        optimizer.setup(model)

        updater = StandardUpdater(train_iter, optimizer, device=gpu)
        evaluator = CEEvaluator(test_iter, model, device=gpu)
    else:
        if args.lossless:
            model = LosslessClassifier(base_model)
        else:
            model = StandardClassifier(base_model)

        train_indice_ranges, train_set, test_indice_ranges, test_set = load_triplet_dataset(args)
        test_triplet, test_labels = zip(*test_set)
        test_triplet = np.asarray(list(test_triplet))
        test_labels = np.asarray(test_labels)

        train_iter = TripletIterator(train_indice_ranges,
                                     train_set,
                                     batch_size=batch_size,
                                     repeat=True,
                                     xp=xp)
        test_iter = TripletIterator(test_indice_ranges,
                                    test_set,
                                    batch_size=batch_size,
                                    xp=xp)

        if int(gpu) >= 0:
            backend.get_device(gpu).use()
            base_model.to_gpu()
            model.to_gpu()

        optimizer = optimizers.Adam(alpha=lr)
        optimizer.setup(model)

        updater = triplet.Updater(train_iter, optimizer, device=gpu)
        evaluator = triplet.Evaluator(test_iter, model, device=gpu)

    if args.eval_only is None:
        trainer = get_trainer(updater, evaluator, epochs)
        if plot_loss:
            trainer.extend(extensions.PlotReport(["main/loss", "validation/main/loss"], "epoch",
                                                 file_name=f"{model_name}_loss.png"))
        trainer.extend(extensions.snapshot(serializers.save_npz, filename= model_name + "_full_{0.updater.epoch:03d}.npz",
                                           target=model))
        best_model_name = model_name + "_full_best.npz"
        trainer.extend(extensions.snapshot(serializers.save_npz, filename=best_model_name, target=model),
                       trigger=triggers.BestValueTrigger("validation/main/loss", lambda best, new: new < best))

        if not args.ce_classifier:
            cluster_dir = os.path.join(writer.logdir, "cluster_imgs")
            os.makedirs(cluster_dir, exist_ok=True)
            trainer.extend(ClusterPlotter(base_model, test_labels, test_triplet, batch_size, xp, cluster_dir),
                           trigger=(1, "epoch"))

        # trainer.extend(VisualBackprop(test_triplet[0], test_labels[0], base_model, [["visual_backprop_anchors"]], xp), trigger=(1, "epoch"))
        # trainer.extend(VisualBackprop(test_triplet[2], test_labels[2], base_model, [["visual_backprop_anchors"]], xp), trigger=(1, "epoch"))

        trainer.run()

        for file in glob.glob(f"result/{model_name}*"):
            shutil.move(file, writer.logdir)
        best_model_path = os.path.join(writer.logdir, best_model_name)
    else:
        best_model_path = args.eval_only

    #################### Evaluation ########################################

    serializers.load_npz(best_model_path, model)
    if args.ce_classifier:
        metrics = evaluate_ce(model, test, batch_size, label_map, xp)
    else:
        metrics = evaluate_triplet(model, test_triplet, test_labels, batch_size, writer, xp)

    with open(os.path.join(writer.logdir, "metrics.log"), "w") as log_file:
        json.dump(metrics, log_file, indent=4)

    print("Done")
    sys.exit(0)


if __name__ == "__main__":
    main()
