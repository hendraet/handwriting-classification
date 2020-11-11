# Handwriting Classification

This repository contains the implemenation of the methods described in the paper [Handwriting Classification for the Analysis of Art-Historical Documents](https://arxiv.org/abs/2011.02264) (preprint).  
This paper was accepted to [FAPER 2020](https://sites.google.com/view/faper-workshop).

## Installation

For installing the dependencies and running the code `python 3.7` and `pip3` are required.  
Install the necessary dependencies by executing `pip install -r requirements.txt` in the root directory.  
For some of the scripts contained in the `prep` directory, it is necessary to install additional dependencies.
These can be installed by executing `pip install -r requirements.txt` in that directory.

## Datasets

The training requires a labelled set of images that display handwriting.
These images should be resized to a size of `64 x 216px` (height x width).  
If the original images have a different aspect ratio, padding that matches the background of the handwriting should be added.
A script to resize the images of a dataset can be found under `prep/image_processing/resize_images.py`.

Each dataset has to be described by a JSON file.
An example for such a file can be found under `datasets/train_test_file_example.json`.  
Each sample has to be labelled as one of the following types: text, num, alpha_num, plz (used for zip codes) or date.
Additionally, each sample needs the path to the corresponding image and the string, which it represents.

For the experiments in the paper mainly two datasets were used.
The first dataset consists of synthesised handwritten text that contains numbers, words and dates.
The images were synthesised using a modified version the GANwriting model, which can be found [here](https://github.com/hendraet/research-GANwriting/).  
The second version of this dataset adds printed text and two more classes: zip codes and alphanumeric strings.
It is denoted 5CHPT (Five Classes of Handwritten and Printed Text).  
These two datasets can be downloaded [here](https://bartzi.de/research/handwriting_classification).

## Pretrained Models

Pretrained models for the datasets mentioned above can be found on [this](https://bartzi.de/research/handwriting_classification) website.

## Training the Models

The `handwriting_embedding` directory contains the files required for training models from scratch.

### Configuration

The models can be trained by executing `train.py` in the `handwriting_embedding` directory.
The script expects five arguments:

- the location of the config file, which is structured like `train.conf` file
- the name under which the trained models should be saved
- the path of the dataset directory that contains the images of the datasets as well as the dataset descriptions of train and test dataset
- the filename of the JSON file for the train dataset
- the filename of the JSON file for the test dataset

Providing no additional arguments will train a ResNet-18 using triplet loss and classifies the test samples using a naive algorithm based on k-means and knn.
Various optional arguments can be provided to change this behaviour.

- `-ll` uses [lossless triplet loss](https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24) instead of the standard triplet loss (not adivsed since it does not perform as well).
- `-llr` uses a classification approach based on log-likelihood ratios instead of the naive algorithm.
- `-ce` trains a Cross-Entropy-loss-based ResNet with a softmax classifier instead of using triplet loss.
This argument cannot be used together with `-ll` or `-llr`.
- `-rs` controls the ResNet size and can be one of the following values: 16, 18, 19, 20, 21, 32, 34, 44, 50, 52, 56, 101, 110, 152.
- `-eo` omits the training step and only evaluates the model that is found under the path that is provided as part of this argument.
- `-ld` specifies the name of the directory in which the model and the results should be logged in the `runs` directory.
If not provided, the standard Tensorboard naming scheme will be used for naming the logging directory.

### Results

Every training or evaluation produces the file `metrics.log`, which is saved in the logging directoy.  
It contains accuracy, precision, recall, f1_scores, the confusion matrix, as well as the initial support for each class.
Precision, recall and F1 score are calculated for each class, as well as the weighted (prefix `w_`) and unweighted (prefix `uw_`) average.
The scores in the lists correspond to the alphabetically sorted list of class tags.

When training a triplet loss model, the embeddings of the test samples are saved as well and can be visualised via Tensorboard.
A two-dimensional visualisation of the embeddings of each epoch can also be found in the `cluster_imgs` directory.

The loss curves are plotted and saved as `[model name]_loss.png`.

Finally, the model is saved after each epoch (`[model name]_full_[epoch].npz`) and the best model is saved separately in `[model name]_full_best.npz`.
The best model is chosen based on the best validation loss.

## Additional Scripts for Dataset Preparation

The `prep` directory contains a set of scripts that were mainly used to create and process datasets.

The `dataset_processing` directory contains scripts for:

- Merging multiple datasets and splitting them into train and test files
- Generating JSON dataset descriptions (in the format of train and test files) based on image file names
- Generating the dataset descriptions for the IAMDB and IAM-Hist-DB datasets
- Creating samples of printed text (as they are used in the 5CHPT dataset)

The `experiments` directory contains a script for automatically repeating experiments and converting the `metrics.log` files into Latex tables

The `iamondb` directory contains a script for the manual extraction of strokes from online handwriting data as well as a script that synthesises new dates and/or numbers out of the extracted strokes.
These scripts were used to generate a dataset for training the GANwriting model.

Scripts for image augmentation, binarisation, colour inversion and resizing can be found in the `image_processing` directory.

Not all of the scripts can be configured via command line arguments and sometimes variables have to be altered in the code to adapt the behaviour.

## Citation

If you find this code useful for your research, please cite our paper:

```
@misc{bartz2020handwriting,
      title={Handwriting Classification for the Analysis of Art-Historical Documents},
      author={Christian Bartz and Hendrik Rätz and Christoph Meinel},
      year={2020},
      eprint={2011.02264},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

