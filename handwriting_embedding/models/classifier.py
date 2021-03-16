import chainer.links as L
from chainer import Chain, report
from chainer import functions as F


def lossless_triplet_loss(anchor, positive, negative, N, beta=None, epsilon=1e-8):
    """
    N  --  The number of dimension
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)
    """

    if beta is None:
        beta = N

    # pos_dist = F.sum((anchor - positive) ** 2, axis=1)
    # neg_dist = F.sum((anchor - negative) ** 2, axis=1)

    pos_dist = -F.log(-(F.sum((anchor - positive) ** 2, axis=1) / beta) + 1 + epsilon)
    neg_dist = -F.log(-((N - F.sum((anchor - negative) ** 2, axis=1)) / beta) + 1 + epsilon)

    loss = pos_dist + neg_dist

    import chainer
    import numpy
    nloss = chainer.as_array(loss)
    if numpy.isnan(nloss).any():
        print()

    return loss


class LosslessClassifier(Chain):
    def __init__(self, predictor):
        super(LosslessClassifier, self).__init__(predictor=predictor)

    def __call__(self, x_a, x_p, x_n):
        y_a, y_p, y_n = (self.predictor(x) for x in (x_a, x_p, x_n))
        y_a = F.sigmoid(y_a)
        y_p = F.sigmoid(y_p)
        y_n = F.sigmoid(y_n)

        N = y_a.shape[-1]
        beta = N
        loss = F.sum(lossless_triplet_loss(y_a, y_p, y_n, N, beta))

        report({'loss': loss}, self)
        return loss


class StandardClassifier(Chain):
    def __init__(self, predictor, train_linear_only=False):
        super(StandardClassifier, self).__init__(predictor=predictor)

        with self.init_scope():
            self.train_linear_only = train_linear_only
            self.linear = L.Linear(None, 512)

    def __call__(self, x_a, x_p, x_n):
        h_a, h_p, h_n = (self.predictor(x) for x in (x_a, x_p, x_n))
        if self.train_linear_only:
            h_a.unchain()
            h_p.unchain()
            h_n.unchain()
        y_a, y_p, y_n = (self.linear(h) for h in (h_a, h_p, h_n))
        loss = F.triplet(y_a, y_p, y_n, margin=1)
        report({'loss': loss}, self)
        return loss


class CrossEntropyClassifier(Chain):
    def __init__(self, predictor, num_classes):
        super(CrossEntropyClassifier, self).__init__(predictor=predictor)

        with self.init_scope():
            self.linear = L.Linear(None, num_classes)

    def __call__(self, x, y):
        h = self.predictor(x)
        h = self.linear(h)
        loss = F.softmax_cross_entropy(h, y)
        report({'loss': loss}, self)
        return loss

    def predict(self, x, return_confidence=False):
        h = self.predictor(x)
        h = self.linear(h)
        softmax_array = F.softmax(h).array
        prediction = self.xp.argmax(softmax_array, axis=1)
        if return_confidence:
            confidence = self.xp.max(softmax_array, axis=1)
            return prediction, confidence
        else:
            return prediction


