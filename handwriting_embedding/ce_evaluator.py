import chainer
import copy
import numpy as np
from chainer import reporter as reporter_module, variable
from chainer.training.extensions import Evaluator


class CEEvaluator(Evaluator):
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
                x, y = zip(*batches)
                x_v = variable.Variable(np.asarray(x))
                x_v.to_gpu()
                y_v = variable.Variable(np.asarray(y))
                y_v.to_gpu()
                eval_func(x_v, y_v)

            summary.add(observation)

        return summary.compute_mean()


