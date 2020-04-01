import pickle

import chainer
import matplotlib
import numpy as np
from chainer import cuda, training
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def remove_black_rect(img):
    tmp = img[:, :, 0]  # only ok if image really grayscale
    summed_columns = np.sum(tmp, axis=0)
    first_col = np.searchsorted(summed_columns, 0, side='right')
    last_col = np.searchsorted(summed_columns[::-1], 0, side='right')

    if (first_col + last_col) < len(summed_columns):
        return img[:, first_col:-(last_col + 1), :]
    else:
        return img


def get_pca(dataset, model, xp):
    with chainer.using_device(model.device):
        embeddings = []
        batch_size = 128  # TODO: remove magic number

        for i in range(0, len(dataset), batch_size):
            batch = xp.array(list(dataset[i:i + batch_size]))
            embedding_batch = model(batch)
            embedding_flat = cuda.to_cpu(embedding_batch.array)
            embeddings.extend(embedding_flat)
            print('.', end='')
        print()

    X = np.array(embeddings)
    pca = PCA(n_components=2)
    fitted_data = pca.fit_transform(X)
    x = fitted_data[:, 0]
    y = fitted_data[:, 1]
    return x, y


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


def draw_embeddings_cluster_with_images(filename, model, labels, dataset, xp, draw_images):
    x, y = get_pca(dataset, model, xp)
    plt.clf()

    fig, ax = plt.subplots()
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

            ax.scatter(coords[:, 0], coords[:, 1], label=item)

    ax.legend(fontsize='xx-small')
    plt.savefig('result/' + filename, dpi=600)
    plt.close(fig)


class ClusterPlotter(training.Extension):
    def __init__(self, model, labels, dataset, xp):
        self._model = model
        self._labels = labels
        self._dataset = dataset
        self._xp = xp

    def __call__(self, trainer):
        epoch = trainer.updater.epoch
        draw_embeddings_cluster_with_images('cluster_epoch_{}.png'.format(epoch), self._model, self._labels,
                                            self._dataset, self._xp, draw_images=False)
