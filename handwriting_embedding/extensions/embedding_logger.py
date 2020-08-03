from chainer import training
from tensorboardX import SummaryWriter

# TODO: is probably not needed
class EmbeddingLogger(training.Extension):
    def __init__(self, model, labels, dataset, xp):
        self._model = model
        self._labels = labels
        self._dataset = dataset
        self._xp = xp
        self.writer = SummaryWriter()

    def __call__(self, trainer):
        pass
