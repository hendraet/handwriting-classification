import argparse
import configparser
import glob
import json
import math
import statistics
import sys

import matplotlib
import numpy as np
import os
import shutil
from chainer import training, cuda, backend, serializers, optimizers
from chainer.iterators import SerialIterator
from chainer.training import extensions, StandardUpdater, triggers
from tensorboardX import SummaryWriter

from dataset_utils import load_triplet_dataset, load_dataset
from evaluation.ce_evaluator import CEEvaluator
from evaluation.classification import evaluate_embeddings, get_metrics, format_metrics
from evaluation.eval_utils import get_embeddings, create_tensorboard_embeddings
from evaluation.log_likelihood_ratio import calc_llr
from extensions.cluster_plotter import ClusterPlotter, draw_embeddings_cluster_with_images
from models.classifier import CrossEntropyClassifier, LosslessClassifier, StandardClassifier
from models.resnet import PooledResNet
from triplet_loss_utils import triplet
from triplet_loss_utils.triplet_iterator import TripletIterator

matplotlib.use("Agg")


def get_trainer(updater, evaluator, epochs):
    trainer = training.Trainer(updater, (epochs, "epoch"), out="result")
    trainer.extend(evaluator)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar((epochs, "epoch"), update_interval=10))
    trainer.extend(extensions.PrintReport(["epoch", "main/loss", "validation/main/loss"]))
    return trainer


def evaluate_triplet(model, train_samples, train_labels, test_samples, test_labels, batch_size, writer, xp):
    test_embeddings = get_embeddings(model.predictor, test_samples, batch_size, xp)
    draw_embeddings_cluster_with_images("cluster_final.png", test_embeddings, test_labels, test_samples,
                                        draw_images=False)
    # draw_embeddings_cluster_with_images("cluster_final_with_images.png", test_embeddings, test_labels, test_triplet,
    #                                     draw_images=True)

    # Add embeddings to projector
    test_samples = 1 - test_samples  # colours are inverted for model - "re-invert" for better visualisation
    create_tensorboard_embeddings(test_samples, test_labels, test_embeddings, writer)

    train_embeddings = get_embeddings(model.predictor, train_samples, batch_size, xp)
    all_metrics = []
    num_runs = 101
    for i in range(num_runs):
        metrics = evaluate_embeddings(train_embeddings, train_labels, test_embeddings, test_labels, verbose=False)
        all_metrics.append(metrics)

    final_metrics = average_all_metrics(all_metrics)
    return final_metrics


def average_all_metrics(all_metrics):
    all_f_scores = [metrics["w_f_score"] for metrics in all_metrics]
    median_idx = all_f_scores.index(statistics.median(all_f_scores))
    final_metrics = {}
    for k in all_metrics[0].keys():
        final_metrics[k] = [metrics[k] for metrics in all_metrics][median_idx]
    return final_metrics


def evaluate_triplet_with_llr(train_samples, train_labels, test_samples, test_labels, log_dir, model, batch_size, xp):
    train_embeddings = get_embeddings(model.predictor, train_samples, batch_size, xp)
    test_embeddings = get_embeddings(model.predictor, test_samples, batch_size, xp)

    llrs = []
    for i in range(1):
        llrs.append(calc_llr(train_embeddings, train_labels, test_embeddings, test_labels, log_dir=log_dir))

    return average_all_metrics(llrs)


# TODO: move eval stuff to eval files
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
    print(format_metrics(metrics))
    return metrics


def main():
    # TODO: cleanup and move to conf or remove conf
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Config file for Training params such as epochs, batch size, lr, etc.")
    parser.add_argument("model_name", type=str, help="The name under which the models will be saved")
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
    parser.add_argument("-llr", action="store_true",
                        help="Evaluate triplets with log-likehood-ratios instead of kmeans/knn")
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
    gpu = config["TRAINING"]["gpu"]

    xp = cuda.cupy if int(gpu) >= 0 else np

    model_name = args.model_name

    # Init tensorboard writer
    if args.log_dir is not None:
        if args.eval_only is not None:
            log_dir = f"runs/{args.log_dir}_eval"
        else:
            log_dir = f"runs/{args.log_dir}"
        if os.path.exists(log_dir):
            user_input = input("Log dir not empty. Clear log dir? (y/N)")
            if user_input == "y":
                shutil.rmtree(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = SummaryWriter()
        log_dir = writer.logdir

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
        test_iter = SerialIterator(test, batch_size, repeat=False, shuffle=False)

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
        ### load dataset
        train_triplet, train_samples, train_labels, test_triplet, test_samples, test_labels = load_triplet_dataset(args)

        # Decide on triplet loss function; spoiler: lossless sucks
        if args.lossless:
            model = LosslessClassifier(base_model)
        else:
            model = StandardClassifier(base_model)

        ### Initialise triple loss model
        train_iter = TripletIterator(train_triplet, batch_size=batch_size, repeat=True, xp=xp)
        test_iter = TripletIterator(test_triplet, batch_size=batch_size, xp=xp)

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
            trainer.extend(ClusterPlotter(base_model, test_labels, test_samples, batch_size, xp, cluster_dir),
                           trigger=(1, "epoch"))

        # trainer.extend(VisualBackprop(test_triplet[0], test_labels[0], base_model, [["visual_backprop_anchors"]], xp), trigger=(1, "epoch"))
        # trainer.extend(VisualBackprop(test_triplet[2], test_labels[2], base_model, [["visual_backprop_anchors"]], xp), trigger=(1, "epoch"))

        trainer.run()

        # serializers.save_npz(os.path.join(writer.logdir, model_name + "_base.npz"), base_model)

        for file in glob.glob(f"result/{model_name}*"):
            shutil.move(file, writer.logdir)
        best_model_path = os.path.join(writer.logdir, best_model_name)
    else:
        best_model_path = args.eval_only

    #################### Evaluation ########################################

    serializers.load_npz(best_model_path, model)
    if args.ce_classifier:
        metrics = evaluate_ce(model, test, batch_size, label_map, xp)
    elif args.llr:
        metrics = evaluate_triplet_with_llr(train_samples, train_labels, test_samples, test_labels, log_dir, model,
                                            batch_size, xp)
    else:
        metrics = evaluate_triplet(model, train_samples, train_labels, test_samples, test_labels, batch_size, writer, xp)

    with open(os.path.join(writer.logdir, "metrics.log"), "w") as log_file:
        json.dump(metrics, log_file, indent=4)

    print("Done")
    # sys.exit(0)
    os._exit(0)


if __name__ == "__main__":
    main()
