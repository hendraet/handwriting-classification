# Handwriting Classification

The `handwriting_embedding` directory contains the proposed models used for handwriting classification.
The `prep` directory contains a set of scripts that were used to create datasets.

## Installation

Install the necessary dependencies by executing `pip install -r requirements.txt` in the root directory.  
For some of the scripts contained in the `prep` directory, it is necessary to install additional dependencies.
These can be installed by executing `pip install -r requirements.txt` in that directory.

## Training the Models

### Configuration

The models can be trained by executing `train.py` in the `handwriting_embedding` directory.
The script expects five arguments:

- the location of the config file, which is structured like `train.conf` file
- the name under which the trained models should be saved
- the path of the dataset directory that contains all the images, as well as the train and test file
- the name of the train file
- the name of the test file

An example for train and test file can be found under `datasets/train_test_file_example.json`.   
These files have to be in JSON format and each sample has to be one of the following types: text, num, alpha_num, plz (used for zip codes) or date.
Additionally, each sample needs the path to the corresponding image and the string, which is contained in it.

Providing no additional arguments will train a ResNet-18 using triplet loss and classifies the test samples using a naive algorithm based on k-means and knn.  
Various optional arguments can be provided to change this behaviour.

- `-ll` uses [lossless triplet loss](https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24) instead of the standard triplet loss (not adivsed since it does not perform as well).
- `-llr` uses a classification approach based on log-likelihood ratios instead of the naive algorithm.
- `-ce` trains a Cross-Entropy-loss-based ResNet instead of using triplet loss.
This argument cannot be used together with `-ll` or `-llr`.
- `-rs` controls the ResNet size and can be one of the following values: 16, 18, 19, 20, 21, 32, 34, 44, 50, 52, 56, 101, 110, 152.
- `-eo` omits the training step and only evaluates the model that is found under the path that is provided as part of this parameter.
- `-ld` specifies the name of the dir in which the model and the results should be logged in the `runs` dir.
If not provided, the standard Tensorboard logging directory will be used.

### Results

Every training or evaluation produces the file `metrics.log`, which is saved in the log dir.  
It contains accuracy, precision, recall, f1_scores, the confusion matrix, as well as the initial support for each class.
Precision, recall and F1 score are calculated for each class, as well as the weighted (prefix `w_`) and unweighted (prefix `uw_`) average.  
The scores in a list correspond to the alphabetically sorted list of class tags.

When training a triplet loss model, the embeddings of the test samples are saved as well and can be visualised via Tensorboard.  
A two-dimensional visualisation of the embeddings of each epoch can also be found in the `cluster_imgs` directory.

The loss curves are plotted and saved as `[model name]_loss.png`.

Finally, the model is saved after each epoch (`[model name]_full_[epoch].npz`) and the best model is saved separately in `[model name]_full_best.npz`.
The best model is chosen based on the best validation loss.
