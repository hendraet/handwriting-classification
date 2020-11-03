import os

import matplotlib
import numpy as np
import seaborn
from chainer import training
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA

from evaluation.eval_utils import get_embeddings
from plot_utils import label_dict, colour_dict, configure_plots

matplotlib.use('Agg')
import matplotlib.pyplot as plt

seaborn.set()
final_cluster_size = True
if final_cluster_size:
    lw = configure_plots(plt, size="medium")
else:
    lw = configure_plots(plt, size="small")


def remove_black_rect(img):
    tmp = img[:, :, 0]  # only ok if image really grayscale
    summed_columns = np.sum(tmp, axis=0)
    first_col = np.searchsorted(summed_columns, 0, side='right')
    last_col = np.searchsorted(summed_columns[::-1], 0, side='right')

    if (first_col + last_col) < len(summed_columns):
        return img[:, first_col:-(last_col + 1), :]
    else:
        return img


def get_pca(embeddings, components=2):
    X = embeddings
    pca = PCA(n_components=components)
    fitted_data = pca.fit_transform(X)
    x = fitted_data[:, 0]
    y = fitted_data[:, 1]
    z = fitted_data[:, 2] if components == 3 else None
    return x, y, z


def imscatter(x, y, images, ax=None, zoom=1.0):
    offset_images = [OffsetImage(image, zoom=zoom) for image in images]
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, offset_image in zip(x, y, offset_images):
        ab = AnnotationBbox(offset_image, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def draw_embeddings_cluster_with_images(filename, embeddings, labels, dataset, draw_images):
    x, y, z = get_pca(embeddings)

    fig, ax = plt.subplots()
    fig.set_size_inches([12.8, 9.6])
    label_set = set(labels)
    for item in label_set:
        indices = []
        for idx, label in enumerate(labels):
            if item == label:
                indices.append(idx)
        if len(indices) > 0:
            coords = np.asarray([[x[idx], y[idx]] for idx in indices])

            if draw_images:
                image_dataset = np.transpose(np.asarray(dataset), (0, 2, 3, 1))
                cropped_images = [remove_black_rect(image_dataset[idx]) for idx in indices]
                imscatter(coords[:, 0], coords[:, 1], cropped_images, zoom=0.15, ax=ax)

            ax.scatter(coords[:, 0], coords[:, 1], label=label_dict[item], c=colour_dict[item], s=9 ** 2)

    ax.legend()
    # plt.savefig(filename, dpi=600)
    print(f"Saving clusters to {os.path.join(os.getcwd(), filename)}")
    plt.savefig(filename)
    plt.close(fig)


class ClusterPlotter(training.Extension):
    def __init__(self, model, labels, dataset, batch_size, xp, logdir):
        self._model = model
        self._labels = labels
        self._dataset = dataset
        self._batchsize = batch_size
        self._xp = xp
        self._logdir = logdir

    def __call__(self, trainer):
        epoch = trainer.updater.epoch
        embeddings = get_embeddings(self._model, self._dataset, self._batchsize, self._xp)
        draw_embeddings_cluster_with_images(f'{self._logdir}/cluster_epoch_{str(epoch).zfill(3)}.png', embeddings,
                                            self._labels, self._dataset, draw_images=False)
