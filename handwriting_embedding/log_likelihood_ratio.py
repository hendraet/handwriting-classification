import itertools
import pickle

import matplotlib
import numpy as np
import random
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn import metrics

matplotlib.use("TkAgg")
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_embeddings_per_class(classes, embeddings, labels):
    samples = {}

    strings_filename = 'strings_full_ds_baseline.pickle'
    with open(strings_filename, 'rb') as f:
        saved_strings = list(pickle.load(f))

    for target_class in classes:
        samples[target_class] = np.asarray([(emb, string) for emb, label, string in zip(embeddings, labels, saved_strings) if label == target_class])
    return samples


def get_dists(samples, same_class, split_diff_dists=False):
    class_dists = {}
    for target_class, emb_str_tuples in samples.items():
        embeddings, strings = zip(*emb_str_tuples)
        if same_class:
            upper_tri_indices = np.triu_indices(len(embeddings), k=1)
            dists = cdist(embeddings, embeddings, 'sqeuclidean')[upper_tri_indices]
            class_dists[target_class] = dists

            full_dists = cdist(embeddings, embeddings, 'sqeuclidean')
            zero_idx = np.where(full_dists == 0.0)
            ut_pairs = list(zip(*upper_tri_indices))
            zero_idx_pairs = list(zip(*zero_idx))
            same_pairs = [(strings[x], strings[y], embeddings[x], embeddings[y]) for x, y in zero_idx_pairs if (x, y) in ut_pairs]
            print()
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


def get_mean_dist_for_class(second_class, test_sample, samples, k=5):
    dists = np.zeros((k,))
    second_class_samples = [sample for sample in samples if sample[0] == second_class]
    for i in range(k):
        second_class_sample = random.choice(second_class_samples)[1]
        dists[i] = cdist(np.expand_dims(test_sample, 0), np.expand_dims(second_class_sample, 0), 'sqeuclidean')
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
        # print(f"Assumed class: {assumed_class}")
        same_dist = get_mean_dist_for_class(assumed_class, test_sample, test_samples)
        same_prob = get_probability_for_dist(same_dist, assumed_class, same_class_dist_hists, hist_edges)

        other_class = np.random.choice([c for c in classes if c != assumed_class])  # TODO: for all other classes?
        other_dist = get_mean_dist_for_class(other_class, test_sample, test_samples)
        other_prob = get_probability_for_dist(other_dist, other_class, other_class_dist_hists, hist_edges)
        # nan messes up the ordering of the sorted llrs and will therefore be replaced with -inf
        llr = np.log(same_prob / other_prob) if other_prob != 0.0 else float("-inf")
        log_likelihood_ratios[assumed_class] = llr

        # print(f"Assumed class: {assumed_class}, Other class: {other_class}")
        # print(f"Dists: {same_dist} vs {other_dist}")
        # print(f"Probs: {same_prob} vs {other_prob}")
        # print(f"LLR: {llr}")

    sorted_llrs = sorted([(cl, value) for cl, value in log_likelihood_ratios.items()], key=lambda x: x[1], reverse=True)
    highest_llr = sorted_llrs[0]
    if highest_llr[1] == float("-inf"):
        print("Best llr was -inf. Likely one or more buckets have no value")
    # print(f"Highest LLR: {highest_llr}")
    predicted_class = highest_llr[0]
    return predicted_class


def calc_llr(embeddings, labels, split_diff_dists=False):
    classes = set(labels)
    samples = get_embeddings_per_class(classes, embeddings, labels)
    # contains all intra-class dists, e.g. the dist between two different text embeddings
    same_class_dists = get_dists(samples, same_class=True)
    # contains the dists from one class (key) to all other classes, e.g. dist text - date, text - num
    diff_class_dists = get_dists(samples, same_class=False, split_diff_dists=split_diff_dists)

    # TODO: scores needed for C_llr calc
    # hist_bins = 1000  # TODO: find or calculate a good value
    hist_bins = 1000  # TODO: find or calculate a good value

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

    # same = np.asarray([v for k, v in same_class_dist_hists.items()])
    # padding = np.zeros((1, hist_bins))
    # diff = np.asarray([v for k, v in other_class_dist_hists.items()])
    # merged = np.concatenate((same, padding, diff))
    # merged_flat = np.concatenate((np.expand_dims(same_class_dist_hist_flat, 0), np.expand_dims(diff_class_dist_hist_flat, 0)))

    # TODO: get actual test samples
    test_samples = []
    for cl in classes:
        for i in range(100):  # TODO: increase
            test_samples.append((cl, samples[cl][i]))

    correct_prediction = {}
    for actual_class, test_sample in test_samples:
        # print(f"=======================================Actual class: {actual_class}===================================")
        predicted_class = get_prediction(test_sample, test_samples, classes, same_class_dist_hists,
                                         other_class_dist_hists, hist_edges)
        # print(f"Actual class: {actual_class}, predicted class: {predicted_class} ({highest_llr[1]})")
        if actual_class not in correct_prediction:
            correct_prediction[actual_class] = []
        correct_prediction[actual_class].append(1 if actual_class == predicted_class else 0)

    for cl, predictions in correct_prediction.items():
        acc = np.mean(np.asarray(predictions))
        print(f"Acc for {cl}: {acc}")

    colour_dict = {
        "text": "b",
        "plz": "g",
        "alpha_num": "y",
        "date": "c",
        "num": "m",
    }

    fig, axs = plt.subplots(len(classes), 2)
    plt.subplots_adjust(hspace=0.6)
    for row, cl in zip(axs, classes):
        row[0].set_title(str(cl))
        center = (hist_edges[:-1] + hist_edges[1:]) / 2
        width = 1.0 * (hist_edges[1] - hist_edges[0])
        xx = np.linspace(0, max_dist, hist_bins)

        same_hist = same_class_dist_hists[cl]
        # row.bar(center, same_hist, align='center', width=width)
        same_hist_dist = stats.rv_histogram((same_hist, hist_edges))
        row[0].plot(xx, same_hist_dist.pdf(xx), "r")

        if split_diff_dists:
            for o_cl in other_class_dist_hists[cl].keys():
                other_hist = other_class_dist_hists[cl][o_cl]
                other_hist_dist = stats.rv_histogram((other_hist, hist_edges))
                row[0].plot(xx, other_hist_dist.pdf(xx), colour_dict[o_cl])
        else:
            other_hist = other_class_dist_hists[cl]
            # row.bar(center, other_hist, align='center', width=width)
            other_hist_dist = stats.rv_histogram((other_hist, hist_edges))
            row[0].plot(xx, other_hist_dist.pdf(xx), "b")

        y_true = np.zeros((len(test_samples),))
        y_score = np.zeros((len(test_samples),))
        for i, (actual_class, test_sample) in enumerate(test_samples):
            dist = get_mean_dist_for_class(cl, test_sample, test_samples)
            y_score[i] = dist
            y_true[i] = 0 if actual_class == cl else 1  # TODO: double check

        y_score = y_score / max_dist
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)

        lw = 2
        row[1].plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        row[1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.show()


def main():
    # embeddings_filename = 'embeddings_5classes_9k.npy'
    embeddings_filename = 'embeddings_full_ds_baseline.npy'
    saved_embeddings = np.load(embeddings_filename)
    print("Embeddings loaded")

    # labels_filename = 'labels_5classes_9k.pickle'
    labels_filename = 'labels_full_ds_baseline.pickle'
    with open(labels_filename, 'rb') as f:
        saved_labels = list(pickle.load(f))
    print("Labels loaded")

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

    calc_llr(saved_embeddings, saved_labels)


if __name__ == '__main__':
    main()
