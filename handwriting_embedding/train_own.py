import argparse
import configparser
import json
import sys

import copy
import matplotlib
import numpy as np
import os
import triplet
from chainer import training, cuda, backend, serializers, optimizers
from chainer.training import extensions
from classification import evaluate_embeddings
from dataset_utils import load_dataset
from eval_utils import create_tensorboard_embeddings
from eval_utils import get_embeddings
from extensions.cluster_plotter import ClusterPlotter
from models import Classifier
from models import PooledResNet
from tensorboardX import SummaryWriter
from triplet_iterator import TripletIterator

matplotlib.use("Agg")


def get_trainer(updater, evaluator, epochs):
    trainer = training.Trainer(updater, (epochs, "epoch"), out="result")
    trainer.extend(evaluator)
    # TODO: reduce LR -- how to update every X epochs?
    # trainer.extend(extensions.ExponentialShift("lr", 0.1, target=lr*0.0001))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(
        (epochs, "epoch"), update_interval=10))
    trainer.extend(extensions.PrintReport(
        ["epoch", "main/loss", "validation/main/loss"]))
    return trainer


def main():
    # TODO: cleanup and move to conf
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Config file for Training params such as epochs, batch size, lr, etc.")
    parser.add_argument("model_suffix", type=str, help="Suffix that should be added to the end of the model filename")
    parser.add_argument("dataset_dir", type=str,
                        help="Directory where the images and the dataset description is stored")
    # parser.add_argument("json_files", nargs="+", type=str,
    #                     help="JSON files that contain the string-path-type mapping for each sample")
    parser.add_argument("train_path", type=str, help="path to JSON file containing train set information")
    parser.add_argument("test_path", type=str, help="path to JSON file containing test set information")
    parser.add_argument("-r", "--retrain", action="store_true", help="Model will be trained from scratch")
    parser.add_argument("-md", "--model-dir", type=str, default="models",
                        help="Dir where models will be saved/loaded from")
    parser.add_argument("-rs", "--resnet-size", type=int, default="18", help="Size of the used ResNet model")
    parser.add_argument("-ld", "--log-dir", type=str, help="name of tensorboard logdir")
    args = parser.parse_args()

    ###################### CONFIG ############################
    retrain = args.retrain
    resnet_size = args.resnet_size
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_name = f"res{str(resnet_size)}_{args.model_suffix}"
    base_model = PooledResNet(resnet_size)
    model = Classifier(base_model)
    plot_loss = True

    config = configparser.ConfigParser()
    config.read(args.config)

    batch_size = int(config["TRAINING"]["batch_size"])
    epochs = int(config["TRAINING"]["epochs"])
    lr = float(config["TRAINING"]["lr"])
    # lr_interval = int(config["TRAINING"]["lr_interval"])
    gpu = config["TRAINING"]["gpu"]

    xp = cuda.cupy if int(gpu) >= 0 else np

    if args.log_dir is not None:
        log_dir = os.path.join("runs/", args.log_dir)
        if os.path.exists(log_dir):
            assert not os.listdir(log_dir), "log dir not empty"
        writer = SummaryWriter(log_dir)
    else:
        writer = SummaryWriter()

    print("RETRAIN:", str(retrain), "MODEL_NAME:", model_name, "BATCH_SIZE:", str(batch_size), "EPOCHS:", str(epochs))

    #################### DATASETS ###########################

    train_triplet, train_labels, test_triplet, test_labels = load_dataset(args)

    #################### Train and Save Model ########################################

    if retrain:
        # serializers.load_npz(os.path.join(model_dir, model_name + "_base.npz"), base_model)
        if int(gpu) >= 0:
            backend.get_device(gpu).use()
            base_model.to_gpu()
            model.to_gpu()

        train_iter = TripletIterator(train_triplet,
                                     batch_size=batch_size,
                                     repeat=True,
                                     xp=xp)
        test_iter = TripletIterator(test_triplet,
                                    batch_size=batch_size,
                                    xp=xp)

        optimizer = optimizers.Adam(alpha=lr)
        optimizer.setup(model)
        updater = triplet.Updater(train_iter, optimizer, device=gpu)

        evaluator = triplet.Evaluator(test_iter, model, device=gpu)

        trainer = get_trainer(updater, evaluator, epochs)
        if plot_loss:
            trainer.extend(extensions.PlotReport(["main/loss", "validation/main/loss"], "epoch", file_name="loss.png"))
        trainer.extend(ClusterPlotter(base_model, test_labels, test_triplet, batch_size, xp), trigger=(1, "epoch"))
        # trainer.extend(VisualBackprop(test_triplet[0], test_labels[0], base_model, [["visual_backprop_anchors"]], xp), trigger=(1, "epoch"))
        # trainer.extend(VisualBackprop(test_triplet[2], test_labels[2], base_model, [["visual_backprop_anchors"]], xp), trigger=(1, "epoch"))

        trainer.run()

        serializers.save_npz(os.path.join(model_dir, model_name + "_base.npz"), base_model)
    else:
        serializers.load_npz(os.path.join(model_dir, model_name + "_base.npz"), base_model)
        print("Models loaded")

        if int(gpu) >= 0:
            backend.get_device(gpu).use()
            base_model.to_gpu()
            model.to_gpu()

        embeddings = get_embeddings(base_model, test_triplet, batch_size, xp)
        # draw_embeddings_cluster_with_images("cluster_final.png", embeddings, test_labels, test_triplet,
        #                                     draw_images=False)
        # draw_embeddings_cluster_with_images("cluster_final_with_images.png", embeddings, test_labels, test_triplet,
        #                                     draw_images=True)

        metrics = evaluate_embeddings(embeddings, test_labels)
        with open(os.path.join(writer.logdir, "metrics.log"), "w") as log_file:
            json.dump(metrics, log_file)

        # Add embeddings to projector
        test_triplet = 1 - test_triplet  # colours are inverted for model - "re-invert" for better visualisation
        create_tensorboard_embeddings(test_triplet, test_labels, embeddings, writer)

    print("Done")
    sys.exit(0)


if __name__ == "__main__":
    main()
