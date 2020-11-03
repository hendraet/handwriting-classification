import pickle
import sys

import matplotlib
import numpy as np
import os
import seaborn as seaborn
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn import metrics

from evaluation.classification import get_metrics, format_metrics
from plot_utils import colour_dict, colour_palette, label_dict, configure_plots

matplotlib.use('Agg')
import matplotlib.pyplot as plt

seaborn.set()
side_by_side = False
rocs = True  # When plotting the ROC curves, this should be set to true
if side_by_side:
    lw = configure_plots(plt, size="large")
elif rocs:
    lw = configure_plots(plt, size="medium")
else:
    lw = configure_plots(plt, size="small")


def get_embeddings_per_class(classes, embeddings, labels):
    samples = {}
    for target_class in classes:
        samples[target_class] = np.asarray([emb for emb, label in zip(embeddings, labels) if label == target_class])
    return samples


def get_dists(samples, same_class, split_diff_dists=False):
    class_dists = {}
    for target_class, embeddings in samples.items():
        if same_class:
            upper_tri_indices = np.triu_indices(len(embeddings), k=1)
            dists = cdist(embeddings, embeddings, 'sqeuclidean')[upper_tri_indices]
            class_dists[target_class] = dists[dists != 0.0]
        else:
            if split_diff_dists:
                class_dists[target_class] = {}
                for other_class in samples.keys():
                    if other_class == target_class:
                        continue
                    dists = cdist(embeddings, samples[other_class], 'sqeuclidean')
                    class_dists[target_class][other_class] = dists.ravel()
            else:
                rest = np.concatenate(
                    [samples[other_class] for other_class in samples.keys() if other_class != target_class])
                dists = cdist(embeddings, rest, 'sqeuclidean')
                class_dists[target_class] = dists.ravel()

    return class_dists


def get_mean_dist_for_class(sample_class, sample, train_samples, same_class):
    exp_sample = np.expand_dims(sample, 0)
    if same_class:
        same_class_samples = np.asarray([sample[1] for sample in train_samples
                                         if sample[0] == sample_class
                                         and not np.array_equal(sample[1], sample)])
        dists = cdist(exp_sample, same_class_samples, 'sqeuclidean')
    else:
        other_class_samples = np.asarray([sample[1] for sample in train_samples if sample[0] != sample_class])
        dists = cdist(exp_sample, other_class_samples, 'sqeuclidean')

    mean_dist = np.mean(dists)
    return mean_dist


def get_probability_for_dist(dist, second_class, class_dist_hists, hist_edges):
    bucket_idx = np.argmax(hist_edges > dist) - 1
    class_dist_hist = class_dist_hists[second_class]
    bucket_value = class_dist_hist[bucket_idx]
    bin_width = hist_edges[bucket_idx + 1] - hist_edges[bucket_idx]
    probability = bucket_value * bin_width
    return probability


def get_prediction(test_sample, test_samples, classes, same_class_dist_hists, other_class_dist_hists, hist_edges):
    log_likelihood_ratios = {}
    for assumed_class in classes:
        same_dist = get_mean_dist_for_class(assumed_class, test_sample, test_samples, same_class=True)
        same_prob = get_probability_for_dist(same_dist, assumed_class, same_class_dist_hists, hist_edges)

        other_class = np.random.choice([c for c in classes if c != assumed_class])  # TODO: for all other classes?
        other_dist = get_mean_dist_for_class(assumed_class, test_sample, test_samples, same_class=False)
        other_prob = get_probability_for_dist(other_dist, other_class, other_class_dist_hists, hist_edges)

        # avoid edge cases and division by zero
        if same_prob == 0.0:
            same_prob = sys.float_info.min
        if other_prob == 0.0:
            other_prob = sys.float_info.min
        llr = np.log(same_prob / other_prob)
        log_likelihood_ratios[assumed_class] = llr

    sorted_llrs = sorted([(cl, value) for cl, value in log_likelihood_ratios.items()], key=lambda x: x[1], reverse=True)
    highest_llr = sorted_llrs[0]
    if highest_llr[1] == float("-inf"):
        print("Best llr was -inf. Likely one or more buckets have no value")
    predicted_class = highest_llr[0]
    return predicted_class


def calc_llr(train_embeddings, train_labels, test_embeddings, test_labels, split_diff_dists=False, log_dir=""):
    classes = sorted(list(set(train_labels)))
    samples = get_embeddings_per_class(classes, train_embeddings, train_labels)
    # contains all intra-class dists, e.g. the dist between two different text embeddings
    same_class_dists = get_dists(samples, same_class=True)
    # contains the dists from one class (key) to all other classes, e.g. dist text - date, text - num
    diff_class_dists = get_dists(samples, same_class=False, split_diff_dists=split_diff_dists)

    hist_bins = 100  # TODO: find or calculate a good value

    if split_diff_dists:
        max_dist_same = max([max(dists) for dists in same_class_dists.values()])
        max_dist_diff = max(
            [max([max(dists) for dists in diff_class_dists[cl].values()]) for cl in diff_class_dists.keys()])
        max_dist = max(max_dist_diff, max_dist_same)
    else:
        max_dist = max([max(dists) for x_dists in (same_class_dists, diff_class_dists) for dists in x_dists.values()])
    print("Dists calculated")

    same_class_dist_hists = {}
    for target_class, dists in same_class_dists.items():
        same_class_dist_hists[target_class], hist_edges = np.histogram(dists, bins=hist_bins, range=(0.0, max_dist),
                                                                       density=True)
    other_class_dist_hists = {}
    if split_diff_dists:
        for target_class, dist_dict in diff_class_dists.items():
            other_class_dist_hists[target_class] = {}
            for other_class, dists in dist_dict.items():
                hist = np.histogram(dists, bins=hist_bins, range=(0.0, max_dist), density=True)[0]
                other_class_dist_hists[target_class][other_class] = hist
    else:
        for target_class, dists in diff_class_dists.items():
            other_class_dist_hists[target_class] = np.histogram(dists, bins=hist_bins, range=(0.0, max_dist),
                                                                density=True)[0]
    print("Hists created")

    ################ Prediction ##############################################################################
    test_samples = list(zip(test_labels, test_embeddings))

    actual_labels = []
    predicted_labels = []
    for actual_class, test_sample in test_samples:
        predicted_class = get_prediction(test_sample, test_samples, classes, same_class_dist_hists,
                                         other_class_dist_hists, hist_edges)
        actual_labels.append(actual_class)
        predicted_labels.append(predicted_class)

    model_metrics = get_metrics(predicted_labels, actual_labels, list(set(actual_labels)))
    print(format_metrics(model_metrics))

    ################ Plotting ##############################################################################
    classes = [c for c in classes]
    fig = plt.figure(figsize=[12.8, 9.6])

    ### Hist ###
    for cl in classes:
        plt.xlabel('Distance')
        plt.ylabel('Likelihood')

        # center = (hist_edges[:-1] + hist_edges[1:]) / 2
        # width = 1.0 * (hist_edges[1] - hist_edges[0])
        xx = np.linspace(0, max_dist, hist_bins)

        same_hist = same_class_dist_hists[cl]
        # row.bar(center, same_hist, align='center', width=width)
        same_hist_dist = stats.rv_histogram((same_hist, hist_edges))
        plt.plot(xx, same_hist_dist.pdf(xx), color=colour_palette[4], lw=lw, label="Intra-class Distance Distribution")

        if split_diff_dists:
            for o_cl in other_class_dist_hists[cl].keys():
                other_hist = other_class_dist_hists[cl][o_cl]
                other_hist_dist = stats.rv_histogram((other_hist, hist_edges))
                plt.plot(xx, other_hist_dist.pdf(xx), colour_dict[o_cl])
        else:
            other_hist = other_class_dist_hists[cl]
            # row.bar(center, other_hist, align='center', width=width)
            other_hist_dist = stats.rv_histogram((other_hist, hist_edges))
            plt.plot(xx, other_hist_dist.pdf(xx), color=colour_palette[0], lw=lw,
                     label="Inter-class Distance Distribution")

        plt.legend(loc='best')
        plt.savefig(os.path.join(log_dir, f"hist_{cl}.png"))
        plt.clf()

    ### Combined ROC ###
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')

    for cl in classes:
        y_true = np.zeros((len(test_samples),))
        y_score = np.zeros((len(test_samples),))
        for i, (actual_class, test_sample) in enumerate(test_samples):
            dist = get_mean_dist_for_class(cl, test_sample, list(zip(train_labels, train_embeddings)), same_class=True)
            y_score[i] = dist
            y_true[i] = 0 if actual_class == cl else 1

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)

        plt.plot(fpr, tpr, color=colour_dict[cl], lw=lw, label=f'{label_dict[cl]}')

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc='lower right')
    plt.savefig(os.path.join(log_dir, f"roc.png"))
    plt.clf()

    return model_metrics


def main():
    # TODO: add argument for filenames
    embeddings_filename = 'embeddings_5classes_9k.npy'
    saved_embeddings = np.load(embeddings_filename)
    print("Embeddings loaded")

    labels_filename = 'labels_5classes_9k.pickle'
    with open(labels_filename, 'rb') as f:
        saved_labels = list(pickle.load(f))
    print("Labels loaded")

    test_labels = saved_labels[-100:]
    test_embeddings = saved_embeddings[-100:]
    saved_labels = saved_labels[:400]
    saved_embeddings = saved_embeddings[:400]

    # TODO: remove
    # num_per_class = 10
    # reps = 100
    # num_samples = reps * num_per_class
    # init = np.zeros((num_samples, 1))
    # for i in range(num_per_class):
    #     for j in range(reps):
    #         init[i * reps + j] = (i + 1) + j * 0.001
    #
    # saved_embeddings = np.repeat(init, 512, axis=1)
    # saved_labels = sorted(["1", "2", "3", "4", "5"] * reps)
    #
    # test_embeddings = copy.deepcopy(saved_embeddings)
    # test_labels = copy.deepcopy(saved_labels)
    #
    calc_llr(saved_embeddings, saved_labels, test_embeddings, test_labels, log_dir="result/llrs")


if __name__ == '__main__':
    main()
