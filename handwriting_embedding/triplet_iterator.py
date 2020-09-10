from __future__ import division

import numpy as np
import queue
import threading

from chainer.dataset.iterator import Iterator


def queue_worker(index_queue, batch_queue, dataset, xp):
    while True:
        batch_begin, batch_end = index_queue.get()
        batches = xp.array(dataset[batch_begin:batch_end])

        batch_anc = batches[xp.arange(0, len(batches), 3)]
        batch_pos = batches[xp.arange(1, len(batches), 3)]
        batch_neg = batches[xp.arange(2, len(batches), 3)]

        batch_queue.put((batch_anc, batch_pos, batch_neg))


class TripletIterator(Iterator):
    # def __init__(self, dataset, ground_truth, batch_size, repeat=False, xp=np):
    def __init__(self, indice_ranges, ground_truth, batch_size, repeat=False, xp=np):
        self.indice_ranges = indice_ranges
        self.init_queues(0)  # Only needed for length calculation
        self.current_idx = (None, None, None)

        self.ground_truth = ground_truth
        self.len_data = 3 * len(ground_truth) * (len(self.init_a_queue) - 1) * (len(self.init_n_queue))
        self.batch_size = batch_size
        self.repeat = repeat
        self.xp = xp

        self.current_position = 0
        self.epoch = 0

    def init_queues(self, anchor_range_idx):
        self.current_a_range_idx = anchor_range_idx
        self.init_a_queue = list(reversed(list(range(self.indice_ranges[self.current_a_range_idx][0],
                                                     self.indice_ranges[self.current_a_range_idx][1]))))
        self.init_p_queue = list(reversed(list(range(self.indice_ranges[self.current_a_range_idx][0],
                                                     self.indice_ranges[self.current_a_range_idx][1]))))
        self.init_n_queue = []
        idx = list(range(len(self.indice_ranges)))
        idx.remove(self.current_a_range_idx)
        for i in idx:
            self.init_n_queue.extend((reversed(list(range(self.indice_ranges[i][0], self.indice_ranges[i][1])))))
        assert len(self.init_n_queue) == len(self.init_a_queue) * (len(self.indice_ranges) - 1)

        self.a_queue = self.init_a_queue.copy()
        self.p_queue = self.init_p_queue.copy()
        self.n_queue = self.init_n_queue.copy()

        self.p_queue.pop()  # avoid same sample for anchor and positive

    def increase_idx(self):
        # Init
        if self.current_idx == (None, None, None):
            self.init_queues(0)
            self.current_idx = (self.a_queue.pop(), self.p_queue.pop(), self.n_queue.pop())
            return True

        if self.n_queue:
            new_idx = (self.current_idx[0], self.current_idx[1], self.n_queue.pop())
        else:
            if self.p_queue and not (self.p_queue[-1] == self.current_idx[0] and len(self.p_queue) < 2):
                self.n_queue = self.init_n_queue.copy()
                new_p = self.p_queue.pop()
                if new_p == self.current_idx[0]:
                    new_p = self.p_queue.pop()
                new_idx = (self.current_idx[0], new_p, self.n_queue.pop())
            else:
                if self.a_queue:
                    self.p_queue = self.init_p_queue.copy()
                    self.n_queue = self.init_n_queue.copy()
                    new_a = self.a_queue.pop()
                    new_p = self.p_queue.pop()
                    if new_p == new_a:
                        new_p = self.p_queue.pop()
                    new_idx = (new_a, new_p, self.n_queue.pop())
                else:
                    # switch anchor class
                    if self.current_a_range_idx + 1 < len(self.indice_ranges):
                        self.init_queues(self.current_a_range_idx + 1)
                        new_idx = (self.a_queue.pop(), self.p_queue.pop(), self.n_queue.pop())
                    else:
                        return False

        self.current_idx = new_idx
        return True

    def __next__(self):
        if not self.increase_idx():
            self.current_position = 0
            self.epoch += 1

            if not self.repeat:
                raise StopIteration

            self.current_idx = (None, None, None)
            self.increase_idx()

        # simulate progress for ProgressBar extension
        self.current_position += 3 * self.batch_size

        anchors = []
        positives = []
        negatives = []
        for i in range(self.batch_size):
            anchors.append(self.ground_truth[self.current_idx[0]][0])
            positives.append(self.ground_truth[self.current_idx[1]][0])
            negatives.append(self.ground_truth[self.current_idx[2]][0])
            if not self.increase_idx():
                break

        samples = (self.xp.asarray(anchors), self.xp.asarray(positives), self.xp.asarray(negatives))
        return samples

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self.len_data

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)
