import chainer
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


