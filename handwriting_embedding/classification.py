import pickle
import scipy
from collections import Counter

import matplotlib
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_embeddings_cluster(filename, embeddings, labels, centroids):
    X = np.array(embeddings)
    pca = PCA(n_components=2)
    fitted_data = pca.fit_transform(X)
    x = fitted_data[:, 0]
    y = fitted_data[:, 1]

    centroids_reduced = pca.transform(centroids)

    plt.clf()

    fig, ax = plt.subplots()

    # gather all the indices of one subclass and plot them class by class, so that they are correctly coloured and named
    label_set = set(labels)
    for item in label_set:
        indices = []
        for idx, label in enumerate(labels):
            if item == label:
                indices.append(idx)
        if len(indices) > 0:
            coords = np.asarray([[x[idx], y[idx]] for idx in indices])
            ax.scatter(coords[:, 0], coords[:, 1], label=item)

    ax.plot(centroids_reduced[:, 0], centroids_reduced[:, 1], 'ko', label='Centroids')

    ax.legend(fontsize='xx-small')
    plt.savefig('result/' + filename, dpi=600)


def translate_labels(actual_labels, kmeans_labels, predicted_centroid_labels):
    num_centroids = len(set(kmeans_labels))
    # Iterate over all the kmeans labels and see if the labels match for the assigned centroid
    translated_labels = []
    for label_idx, label in enumerate(kmeans_labels):
        for centroid_idx in range(num_centroids):
            if label == centroid_idx:  # label belongs to this centroid
                actual_centroid_label = predicted_centroid_labels[centroid_idx]
                actual_label = actual_labels[label_idx]
                if actual_centroid_label == actual_label:
                    translated_labels.append(actual_label)
                else:  # wrongly classified
                    translated_labels.append(actual_label + ' as ' + actual_centroid_label)
                break
    return translated_labels


def classify_embeddings(embeddings, support_labels, support_embeddings, possible_classes, actual_labels=None):
    # TODO: support Hannes method
    support_label_count = Counter(support_labels)
    assert len(set(support_label_count.values())) == 1 and len(support_label_count.values()) == len(possible_classes), \
        "All possible classes should have the same amount of support labels."

    min_num_of_support_labels = min(support_label_count.values())
    if min_num_of_support_labels < 5:
        print("There are few support samples (<5), which could lead to inaccurate results. Consider adding more "
              "samples.")

    # TODO: "quality" of support labels can have a large impact on kmeans
    # ideally finds the centroid for each class in test dataset
    num_centroids = len(possible_classes)
    centroid, labels = kmeans2(embeddings, num_centroids, minit='points')

    # labeling the centroids with knn on test embeddings
    knn = KNeighborsClassifier(n_neighbors=min_num_of_support_labels)
    knn.fit(support_embeddings, support_labels)
    predicted_centroid_labels = knn.predict(centroid)  # TODO: maybe also generate probabilities

    if actual_labels is not None:
        translated_labels = translate_labels(actual_labels, labels, predicted_centroid_labels)
        draw_embeddings_cluster('embedding_classification', embeddings, translated_labels, centroid)

    # TODO: add some kind of confidence
    return [predicted_centroid_labels[label] for label in labels]


### Utils for log-likelihood

# first calculate dist histograms on labeled data
# then calculate likelihoods on unlabeled data through calculating log-likelihoods for all classes

def get_embeddings_per_class(classes, embeddings, labels):
    samples = {}
    for target_class in classes:
        samples[target_class] = np.asarray([emb for emb, label in zip(embeddings, labels) if label == target_class])
    return samples


def get_dists(samples, same_class):
    class_dists = {}
    for target_class, embeddings in samples.items():
        if same_class:
            # np.unique to remove the second comparison between two samples because datasets are symetric
            dists = np.unique(cdist(embeddings, embeddings, 'sqeuclidean'))
            class_dists[target_class] = dists[dists != 0]  # remove comparisons of same samples
        else:
            rest = np.concatenate([samples[other_class] for other_class in samples.keys() if other_class != target_class])
            dists = cdist(embeddings, rest, 'sqeuclidean')
            class_dists[target_class] = dists.ravel()

    return class_dists


### Hannes

# def genuine_genuine_dists(data, average):
#     gen_keys = [k for k in data.keys() if 'f' not in k]
#     dists = {}
#     for k in gen_keys:
#         gen = cuda.cupy.asnumpy(data[k])
#         if average:
#             gen_mean = []
#             for i in range(len(gen)):
#                 others = list(range(len(gen)))
#                 others.remove(i)
#                 # choose NUM_REF of others for reference
#                 # fails (and should fail) if len(others) < NUM_REF
#                 others = np.random.choice(others, replace=False, size=NUM_REF)
#                 gen_mean.append(np.mean(gen[others], axis=0))
#             dists[k] = cdist(gen_mean, gen, 'sqeuclidean')
#         else:
#             d = np.unique(cdist(gen, gen, 'sqeuclidean'))
#             dists[k] = d[d != 0]  # remove same sample comparisons
#             # dists[k] = cdist(gen, gen, DIST_METHOD)
#     return dists
#
#
# def genuine_forgeries_dists(data, average):
#     gen_keys = [k for k in data.keys() if 'f' not in k]
#     dists = {}
#     for k in gen_keys:
#         gen = cuda.cupy.asnumpy(data[k])
#         if average:
#             gen_mean = []
#             for i in range(len(gen)):
#                 others = list(range(len(gen)))
#                 others.remove(i)
#                 gen_mean.append(np.mean(gen[others], axis=0))
#             gen = gen_mean
#         forge = cuda.cupy.asnumpy(data[k + '_f'])
#         # np.random.shuffle(forge)
#         # forge = forge[:5]  # HACK reduce number of forgeries
#         dists[k] = cdist(gen, forge, 'sqeuclidean')
#     return dists


def dist_to_score(dist, max_dist):
    """Supposed to compute P(target_trial | s)"""
    return max(0, 2.5 * dist / max_dist)
    # return max(0, 1 - dist / max_dist)


def get_probability_for_class(test_sample, samples, class_dist_hists, second_class, hist_edges):
    rand_idx = np.random.randint(len(samples[second_class]))
    second_class_sample = samples[second_class][rand_idx]
    class_dist_hist = class_dist_hists[second_class]
    dist = cdist(np.expand_dims(test_sample, 0), np.expand_dims(second_class_sample, 0), 'sqeuclidean')

    bucket_idx = np.argmax(hist_edges > dist) - 1
    bucket_value = class_dist_hist[bucket_idx]
    bin_width = hist_edges[bucket_idx + 1] - hist_edges[bucket_idx]
    return bucket_value * bin_width


# TODO: rename
def blah(embeddings, labels):
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
        same_class_dist_hists[target_class], hist_edges = np.histogram(dists, bins=hist_bins, range=(0.0, max_dist), density=True)

    other_class_dist_hists = {}
    for target_class, dists in diff_class_dists.items():
        other_class_dist_hists[target_class] = np.histogram(dists, bins=hist_bins, range=(0.0, max_dist), density=True)[0]

    # same = np.asarray([v for k, v in same_class_dist_hists.items()])
    # padding = np.zeros((1, hist_bins))
    # diff = np.asarray([v for k, v in other_class_dist_hists.items()])
    # merged = np.concatenate((same, padding, diff))
    # merged_flat = np.concatenate((np.expand_dims(same_class_dist_hist_flat, 0), np.expand_dims(diff_class_dist_hist_flat, 0)))

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(classes), 1)
    for ax, cl in zip(axs, classes):
        center = (hist_edges[:-1] + hist_edges[1:]) / 2
        width = 1.0 * (hist_edges[1] - hist_edges[0])
        xx = np.linspace(0, max_dist, hist_bins)

        same_hist = same_class_dist_hists[cl]
        # ax.bar(center, same_hist, align='center', width=width)
        same_hist_dist = scipy.stats.rv_histogram((same_hist, hist_edges))
        ax.plot(xx, same_hist_dist.pdf(xx), "r")

        other_hist = other_class_dist_hists[cl]
        # ax.bar(center, other_hist, align='center', width=width)
        other_hist_dist = scipy.stats.rv_histogram((other_hist, hist_edges))
        ax.plot(xx, other_hist_dist.pdf(xx), "b")

    plt.show()

    # TODO
    return

    test_samples = []
    for cl in classes:
        for i in range(1000):
            test_samples.append((cl, samples[cl][i]))

    correct_prediction = {}
    for actual_class, test_sample in test_samples:
        # TODO: method?
        log_likelihood_ratios = {}
        for assumed_class in classes:
            same_prob = get_probability_for_class(test_sample, samples, same_class_dist_hists, assumed_class, hist_edges)

            other_class = np.random.choice([c for c in classes if c != assumed_class])
            other_prob = get_probability_for_class(test_sample, samples, other_class_dist_hists, other_class, hist_edges)
            llr = np.log(same_prob / other_prob)
            log_likelihood_ratios[assumed_class] = llr

        highest_llr = sorted([(cl, value) for cl, value in log_likelihood_ratios.items()], key=lambda x: x[1])[0]
        predicted_class = highest_llr[0]
        # print(f"Actual class: {actual_class}, predicted class: {predicted_class} ({highest_llr[1]})")
        if actual_class not in correct_prediction:
            correct_prediction[actual_class] = []
        correct_prediction[actual_class].append(1 if actual_class == predicted_class else 0)

    for cl, predictions in correct_prediction.items():
        acc = np.mean(np.asarray(predictions))
        print(f"Acc for {cl}: {acc}")

    return


def main():
    embeddings_filename = 'embeddings_5classes_9k.npy'
    labels_filename = 'labels_5classes_9k.pickle'
    saved_embeddings = np.load(embeddings_filename)
    with open(labels_filename, 'rb') as f:
        saved_labels = list(pickle.load(f))

    # Mock more data
    # saved_embeddings = np.append(saved_embeddings, [(saved_embeddings[0] + saved_embeddings[1]) / 2], axis=0)
    # saved_embeddings = np.append(saved_embeddings, [(saved_embeddings[1] + saved_embeddings[2]) / 2], axis=0)
    # saved_labels.append('blah')
    # saved_labels.append('blah')

    # indices = [0, 1, 2, 3, 4, 5, 6, 8, 11, 14]
    # support_labels = np.take(saved_labels, indices)
    # support_embeddings = np.take(saved_embeddings, indices, axis=0)

    # arguable if test dataset can be part of the embeddings or not
    # saved_labels = np.delete(saved_labels, indices, axis=0)
    # saved_embeddings = np.delete(saved_embeddings, indices, axis=0)

    # TODO
    support_labels = saved_labels
    support_embeddings = saved_embeddings

    blah(support_embeddings, support_labels)
    return

    predicted_labels = classify_embeddings(saved_embeddings, support_labels, support_embeddings, saved_labels)

    accuracy = accuracy_score(saved_labels, predicted_labels)
    precision, recall, f_score, support = precision_recall_fscore_support(saved_labels, predicted_labels)

    print(f"Results:\n"
          f"Accuracy:  {accuracy}\n"
          f"Precision: {precision}\n"
          f"Recall:    {recall}\n"
          f"F-score:   {f_score}\n"
          f"Support:   {support}")


if __name__ == '__main__':
    main()
