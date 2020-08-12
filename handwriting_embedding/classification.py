import pickle
from collections import Counter

import argparse
import matplotlib
import numpy as np
import random
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use("TkAgg")
# matplotlib.use('Agg')
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
    support_label_count = Counter(support_labels)
    assert len(set(support_label_count.values())) == 1 and len(support_label_count.values()) == len(possible_classes), \
        "All possible classes should have the same amount of support labels."

    min_num_of_support_labels = min(support_label_count.values())
    if min_num_of_support_labels < 5:
        print("There are few support samples (<5), which could lead to inaccurate results. Consider adding more "
              "samples.")

    # ideally finds the centroid for each class in test dataset
    num_centroids = len(possible_classes)
    centroid, labels = kmeans2(embeddings, num_centroids, minit='points')

    # labeling the centroids with knn on test embeddings
    knn = KNeighborsClassifier(n_neighbors=min_num_of_support_labels)
    knn.fit(support_embeddings, support_labels)
    predicted_centroid_labels = knn.predict(centroid)
    if len(set(predicted_centroid_labels)) < num_centroids:
        knn_probs = knn.predict_proba(centroid)
        print

    if actual_labels is not None:
        translated_labels = translate_labels(actual_labels, labels, predicted_centroid_labels)
        draw_embeddings_cluster('embedding_classification', embeddings, translated_labels, centroid)

    return [predicted_centroid_labels[label] for label in labels]


def get_datasets(saved_embeddings, saved_labels, ratio=0.1, stable=False):
    min_num_of_samples = min(Counter(saved_labels).values())
    num_support_samples = int(ratio * min_num_of_samples)

    if not stable:
        tmp = list(zip(saved_embeddings, saved_labels))
        random.shuffle(tmp)
        saved_embeddings, saved_labels = zip(*tmp)

    support_samples = {}
    test_embeddings = []
    test_labels = []
    for emb, label in zip(saved_embeddings, saved_labels):
        if label not in support_samples:
            support_samples[label] = []
        if len(support_samples[label]) < num_support_samples:
            support_samples[label].append(emb)
        else:
            test_embeddings.append(emb)
            test_labels.append(label)

    flat_support_samples = []
    for label, embeddings in support_samples.items():
        assert len(embeddings) == num_support_samples
        for emb in embeddings:
            flat_support_samples.append((emb, label))
    support_embeddings, support_labels = zip(*flat_support_samples)
    support_embeddings = np.asarray(support_embeddings)

    return support_embeddings, support_labels, test_embeddings, test_labels


def evaluate_dataset(saved_labels, support_embeddings, support_labels, test_embeddings, test_labels, verbose=True):
    classes = list(set(saved_labels))
    # predicted_labels = classify_embeddings(test_embeddings, support_labels, support_embeddings, classes,
    #                                        actual_labels=test_labels)
    predicted_labels = classify_embeddings(test_embeddings, support_labels, support_embeddings, classes)

    accuracy = accuracy_score(test_labels, predicted_labels)
    precision, recall, f_score, support = precision_recall_fscore_support(test_labels, predicted_labels)
    w_precision, w_recall, w_f_score, _ = precision_recall_fscore_support(test_labels, predicted_labels,
                                                                          average="weighted")
    if verbose:
        print(set(predicted_labels))
        print(f"Predicted label distribution: {Counter(predicted_labels)}")
        print(f"Results:\n"
              f"Accuracy:  {accuracy}\n"
              f"Classes:    {''.join(el.ljust(11) for el in [*sorted(classes), ' weighted'])}\n"
              f"Precision: {precision} {w_precision}\n"
              f"Recall:    {recall} {w_recall}\n"
              f"F-score:   {f_score} {w_f_score}\n"
              f"Support:    {''.join(str(el).ljust(11) for el in support)}")

    return w_f_score


def run_experiment(saved_embeddings, saved_labels):
    num_samples = len(saved_labels)
    fig, ax = plt.subplots()

    weighted_fscores = []
    support_ratios = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7]
    for i, ratio in enumerate(support_ratios):
        print(f"Ratio: {ratio}")
        support_embeddings, support_labels, test_embeddings, test_labels = get_datasets(saved_embeddings, saved_labels,
                                                                                        ratio=ratio)
        # blah(support_embeddings, support_labels)
        weighted_fscores.append([])
        for _ in range(25):
            w_f_score = evaluate_dataset(saved_labels, support_embeddings, support_labels, test_embeddings, test_labels,
                                         verbose=False)
            weighted_fscores[i].append(w_f_score)
    ax.boxplot(weighted_fscores, labels=[int(num_samples * r) for r in support_ratios])
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--single-run", action="store_true",
                        help="Run a single experiment instead of having multiple runs with different amount of "
                             "support embedding.")
    args = parser.parse_args()

    embeddings_filename = 'embeddings_5classes_9k.npy'
    labels_filename = 'labels_5classes_9k.pickle'
    saved_embeddings = np.load(embeddings_filename)
    with open(labels_filename, 'rb') as f:
        saved_labels = list(pickle.load(f))

    if args.single_run:
        support_embeddings, support_labels, test_embeddings, test_labels = get_datasets(saved_embeddings, saved_labels)
        w_f_score = evaluate_dataset(saved_labels, support_embeddings, support_labels, test_embeddings, test_labels)
    else:
        run_experiment(saved_embeddings, saved_labels)


if __name__ == '__main__':
    main()
