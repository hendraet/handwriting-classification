import matplotlib
import numpy as np
from chainer import training
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA

from handwriting_embedding.eval_utils import get_embeddings

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


def get_pca(embeddings):
    X = embeddings
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


def draw_embeddings_cluster_with_images(filename, embeddings, labels, dataset, draw_images):
    x, y = get_pca(embeddings)
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
    def __init__(self, model, labels, dataset, batch_size, xp):
        self._model = model
        self._labels = labels
        self._dataset = dataset
        self._batchsize = batch_size
        self._xp = xp

    def __call__(self, trainer):
        epoch = trainer.updater.epoch
        embeddings = get_embeddings(self._model, self._dataset, self._batchsize, self._xp)
        draw_embeddings_cluster_with_images('cluster_epoch_{}.png'.format(str(epoch).zfill(3)), embeddings,
                                            self._labels, self._dataset, draw_images=False)
