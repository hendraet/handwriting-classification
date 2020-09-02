import pickle

import matplotlib
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

matplotlib.use("TkAgg")
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_embeddings_per_class(classes, embeddings, labels):
    samples = {}
    for target_class in classes:
        samples[target_class] = np.asarray([emb for emb, label in zip(embeddings, labels) if label == target_class])
    return samples


def get_dists(samples, same_class):
    class_dists = {}
    for target_class, embeddings in samples.items():
        if same_class:
            upper_tri_indices = np.triu_indices(len(embeddings), k=1)
            # dists = np.unique(cdist(embeddings, embeddings, 'sqeuclidean'))
            dists = cdist(embeddings, embeddings, 'sqeuclidean')[upper_tri_indices]
            # class_dists[target_class] = dists[dists != 0]  # remove comparisons of same samples
            class_dists[target_class] = dists
        else:
            rest = np.concatenate(
                [samples[other_class] for other_class in samples.keys() if other_class != target_class])
            dists = cdist(embeddings, rest, 'sqeuclidean')
            class_dists[target_class] = dists.ravel()

    return class_dists


def calc_llr(embeddings, labels):
    classes = set(labels)
    samples = get_embeddings_per_class(classes, embeddings, labels)
    # contains all intra-class dists, e.g. the dist between two different text embeddings
    same_class_dists = get_dists(samples, same_class=True)
    # contains the dists from one class (key) to all other classes, e.g. dist text - date, text - num
    diff_class_dists = get_dists(samples, same_class=False)

    # TODO: scores needed for C_llr calc
    hist_bins = 1000  # TODO: find or calculate a good value
    max_dist = max([max(dists) for x_dists in (same_class_dists, diff_class_dists) for dists in x_dists.values()])

    same_class_dist_hists = {}
    for target_class, dists in same_class_dists.items():
        same_class_dist_hists[target_class], hist_edges = np.histogram(dists, bins=hist_bins, range=(0.0, max_dist),
                                                                       density=True)
    other_class_dist_hists = {}
    for target_class, dists in diff_class_dists.items():
        other_class_dist_hists[target_class] = np.histogram(dists, bins=hist_bins, range=(0.0, max_dist), density=True)[
            0]

    # TODO:

    # same = np.asarray([v for k, v in same_class_dist_hists.items()])
    # padding = np.zeros((1, hist_bins))
    # diff = np.asarray([v for k, v in other_class_dist_hists.items()])
    # merged = np.concatenate((same, padding, diff))
    # merged_flat = np.concatenate((np.expand_dims(same_class_dist_hist_flat, 0), np.expand_dims(diff_class_dist_hist_flat, 0)))


    fig, axs = plt.subplots(len(classes), 1)
    for ax, cl in zip(axs, classes):
        ax.set_title(str(cl))
        center = (hist_edges[:-1] + hist_edges[1:]) / 2
        width = 1.0 * (hist_edges[1] - hist_edges[0])
        xx = np.linspace(0, max_dist, hist_bins)

        same_hist = same_class_dist_hists[cl]
        # ax.bar(center, same_hist, align='center', width=width)
        same_hist_dist = stats.rv_histogram((same_hist, hist_edges))
        ax.plot(xx, same_hist_dist.pdf(xx), "r")

        other_hist = other_class_dist_hists[cl]
        # ax.bar(center, other_hist, align='center', width=width)
        other_hist_dist = stats.rv_histogram((other_hist, hist_edges))
        ax.plot(xx, other_hist_dist.pdf(xx), "b")

    plt.show()

    # TODO
    # test_samples = []
    # for cl in classes:
    #     for i in range(1000):
    #         test_samples.append((cl, samples[cl][i]))
    #
    # correct_prediction = {}
    # for actual_class, test_sample in test_samples:
    #     # TODO: method?
    #     log_likelihood_ratios = {}
    #     for assumed_class in classes:
    #         same_prob = get_probability_for_class(test_sample, samples, same_class_dist_hists, assumed_class,
    #                                               hist_edges)
    #
    #         other_class = np.random.choice([c for c in classes if c != assumed_class])
    #         other_prob = get_probability_for_class(test_sample, samples, other_class_dist_hists, other_class,
    #                                                hist_edges)
    #         llr = np.log(same_prob / other_prob)
    #         log_likelihood_ratios[assumed_class] = llr
    #
    #     highest_llr = sorted([(cl, value) for cl, value in log_likelihood_ratios.items()], key=lambda x: x[1])[0]
    #     predicted_class = highest_llr[0]
    #     # print(f"Actual class: {actual_class}, predicted class: {predicted_class} ({highest_llr[1]})")
    #     if actual_class not in correct_prediction:
    #         correct_prediction[actual_class] = []
    #     correct_prediction[actual_class].append(1 if actual_class == predicted_class else 0)
    #
    # for cl, predictions in correct_prediction.items():
    #     acc = np.mean(np.asarray(predictions))
    #     print(f"Acc for {cl}: {acc}")
    #
    # return


def main():
    embeddings_filename = 'embeddings_5classes_9k.npy'
    saved_embeddings = np.load(embeddings_filename)

    labels_filename = 'labels_5classes_9k.pickle'
    with open(labels_filename, 'rb') as f:
        saved_labels = list(pickle.load(f))

    # saved_embeddings = np.array([[1], [2], [3], [4], [5]])
    # saved_embeddings = np.repeat(saved_embeddings, 10, axis=0)
    # saved_embeddings = np.repeat(saved_embeddings, 512, axis=1)
    #
    # saved_labels = sorted(["1", "2", "3", "4", "5"] * 10)

    calc_llr(saved_embeddings, saved_labels)


if __name__ == '__main__':
    main()
