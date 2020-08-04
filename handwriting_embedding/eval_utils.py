from math import sqrt

import chainer
import cv2
import numpy as np
from chainer import cuda


def get_embeddings(model, dataset, batch_size, xp):
    with chainer.using_device(model.device):
        embeddings = []
        for i in range(0, len(dataset), batch_size):
            batch = xp.array(list(dataset[i:i + batch_size]))
            embedding_batch = model(batch)
            embedding_flat = cuda.to_cpu(embedding_batch.array)
            embeddings.extend(embedding_flat)

    return np.array(embeddings)


def create_tensorboard_embeddings(test_triplet, test_labels, embeddings, writer):
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


