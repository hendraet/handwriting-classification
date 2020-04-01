"""
This module contains support classes for dealing with triplets, mostly
adjusting or replacing Chainer's built-in utilities.
"""
import copy

import chainer
from chainer import reporter as reporter_module
from chainer import variable

from chainer.training.extensions import Evaluator
from chainer.training.updater import StandardUpdater


class Updater(StandardUpdater):
    """
    Just a StandardUpdater, with it's update_core function adjusted to work
    with the triplet data I feed it.
    """

    def update_core(self):
        with chainer.using_device(self.device):
            self.update_network()

    def update_network(self):
        batches = self._iterators['main'].next()

        in_vars = (variable.Variable(batch) for batch in batches)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        optimizer.update(loss_func, *in_vars)


class Evaluator(Evaluator):
    """
    An Evaluator for triplet data.
    """

    def evaluate(self):
        with chainer.using_device(self.device):
            return self.evaluate_core()

    def evaluate_core(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for batches in it:
            observation = {}
            with reporter_module.report_scope(observation), chainer.no_backprop_mode():
                in_vars = [variable.Variable(batch) for batch in batches]
                eval_func(*in_vars)

            summary.add(observation)

        return summary.compute_mean()
