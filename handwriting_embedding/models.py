from chainer import Chain, report
from chainer import functions as F
from chainer.links.model.vision.resnet import _global_average_pooling_2d

from resnet import ResNet


class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x_a, x_p, x_n):
        y_a, y_p, y_n = (self.predictor(x) for x in (x_a, x_p, x_n))
        loss = F.triplet(y_a, y_p, y_n, margin=1)
        report({'loss': loss}, self)
        return loss


class PooledResNet(Chain):
    def __init__(self, n_layers):
        super(PooledResNet, self).__init__()

        with self.init_scope():
            self.feature_extractor = ResNet(n_layers)
        self.visual_backprop_anchors = []

    def __call__(self, x):
        self.visual_backprop_anchors.clear()
        h = self.feature_extractor(x)
        self.visual_backprop_anchors.append(h)
        h = _global_average_pooling_2d(h)
        return h

