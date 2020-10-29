import argparse
import json
import pickle
import warnings
from collections import Counter

import copy
import matplotlib
import numpy as np
import random
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
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
    support_label_count = Counter(support_labels)
    assert len(set(support_label_count.values())) == 1 and len(support_label_count.values()) == len(possible_classes), \
        "All possible classes should have the same amount of support labels."

    min_num_of_support_labels = min(support_label_count.values())

    # ideally finds the centroid for each class in test dataset
    num_centroids = len(possible_classes)
    centroid, labels = kmeans2(embeddings, num_centroids, minit='points')

    # labeling the centroids with knn on test embeddings
    knn = KNeighborsClassifier(n_neighbors=min_num_of_support_labels)
    knn.fit(support_embeddings, support_labels)
    predicted_centroid_labels = knn.predict(centroid)
    if len(set(predicted_centroid_labels)) < num_centroids:
        knn_probs = knn.predict_proba(centroid)

    if actual_labels is not None:
        translated_labels = translate_labels(actual_labels, labels, predicted_centroid_labels)
        draw_embeddings_cluster('embedding_classification', embeddings, translated_labels, centroid)

    return [predicted_centroid_labels[label] for label in labels]


def get_datasets(saved_embeddings, saved_labels, ratio=0.1, stable=False):
    embeddings = saved_embeddings.copy()
    labels = copy.copy(saved_labels)

    min_num_of_samples = min(Counter(labels).values())
    num_support_samples = int(ratio * min_num_of_samples)

    if not stable:
        tmp = list(zip(embeddings, labels))
        random.shuffle(tmp)
        embeddings, labels = zip(*tmp)

    support_samples = {}
    test_embeddings = []
    test_labels = []
    for emb, label in zip(embeddings, labels):
        if label not in support_samples:
            support_samples[label] = []
        if len(support_samples[label]) < num_support_samples:
            support_samples[label].append(emb)
        else:
            test_embeddings.append(emb)
            test_labels.append(label)

    flat_support_samples = []
    for label, embedding_list in support_samples.items():
        assert len(embedding_list) == num_support_samples
        for emb in embedding_list:
            flat_support_samples.append((emb, label))
    support_embeddings, support_labels = zip(*flat_support_samples)
    support_embeddings = np.asarray(support_embeddings)

    return support_embeddings, support_labels, test_embeddings, test_labels


def format_metrics(metrics):
    formatted_metrics = f"Results:\n" \
                        f"Classes:   {''.join(el.ljust(20) for el in [*sorted(metrics['classes']), ' weighted', 'unweighted'])}\n" \
                        f"Precision: {''.join(str(el).ljust(20) for el in metrics['precision'])} {metrics['w_precision']} {metrics['uw_precision']}\n" \
                        f"Recall:    {''.join(str(el).ljust(20) for el in metrics['recall'])} {metrics['w_recall']} {metrics['uw_recall']}\n" \
                        f"F-score:   {''.join(str(el).ljust(20) for el in metrics['f_score'])} {metrics['w_f_score']} {metrics['uw_f_score']}\n" \
                        f"Support:   {''.join(str(el).ljust(20) for el in metrics['support'])}\n" \
                        f"Predicted label distribution: {metrics['predicted_distribution']}\n" \
                        f"Accuracy:  {metrics['accuracy']}\n" \
                        f"CM:\n{np.asarray(metrics['confusion_matrix'])}"

    return formatted_metrics


def evaluate_dataset(support_embeddings, support_labels, test_embeddings, test_labels, verbose=True):
    classes = list(set(support_labels))
    predicted_labels = classify_embeddings(test_embeddings, support_labels, support_embeddings, classes)

    metrics = get_metrics(predicted_labels, test_labels, classes)

    if verbose:
        print(format_metrics(metrics))

    return metrics


def get_metrics(predicted_labels, test_labels, classes):
    accuracy = accuracy_score(test_labels, predicted_labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, f_score, support = precision_recall_fscore_support(test_labels, predicted_labels)
        w_precision, w_recall, w_f_score, _ = precision_recall_fscore_support(test_labels, predicted_labels,
                                                                              average="weighted")
        uw_precision, uw_recall, uw_f_score, _ = precision_recall_fscore_support(test_labels, predicted_labels,
                                                                                 average="macro")

    if isinstance(test_labels[0], str):
        cm = confusion_matrix(test_labels, predicted_labels, labels=sorted(classes))
    else:
        cm = confusion_matrix(test_labels, predicted_labels)

    metrics = {
        "accuracy": accuracy,
        "precision": precision.tolist(),
        "w_precision": w_precision,
        "uw_precision": uw_precision,
        "recall": recall.tolist(),
        "w_recall": w_recall,
        "uw_recall": uw_recall,
        "f_score": f_score.tolist(),
        "w_f_score": w_f_score,
        "uw_f_score": uw_f_score,
        "support": support.tolist(),
        "predicted_distribution": Counter(predicted_labels),
        "classes": sorted(classes),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def get_num_of_support_samples(train_embeddings, train_labels, num_support_samples):
    support_samples = {}
    for emb, label in zip(train_embeddings, train_labels):
        if label not in support_samples:
            support_samples[label] = []
        if len(support_samples[label]) < num_support_samples:
            support_samples[label].append(emb)
        else:
            continue

    flat_support_samples = []
    for label, embedding_list in support_samples.items():
        assert len(embedding_list) == num_support_samples
        for emb in embedding_list:
            flat_support_samples.append((emb, label))
    support_embeddings, support_labels = zip(*flat_support_samples)
    support_embeddings = np.asarray(support_embeddings)

    return support_embeddings, support_labels


def run_experiment(saved_embeddings, saved_labels):
    test_labels = saved_labels[-1000:]
    test_embeddings = saved_embeddings[-1000:]
    train_labels = saved_labels[:-1000]
    train_embeddings = saved_embeddings[:-1000]

    fig, ax = plt.subplots()

    num_runs = 1000
    weighted_fscores = []
    support_sample_sizes = [25, 27, 30, 32, 35, 37, 40, 42, 45, 47, 50, 55]
    for i, num_support_samples in enumerate(support_sample_sizes):
        print(f"Ratio: {num_support_samples}")
        support_embeddings, support_labels = get_num_of_support_samples(train_embeddings, train_labels,
                                                                        num_support_samples)
        weighted_fscores.append([])
        for j in range(num_runs):
            if (j + 1) % 10 == 0:
                print(f"Run {j + 1} of {num_runs}")
            metrics = evaluate_dataset(support_embeddings, support_labels, test_embeddings, test_labels,
                                       verbose=False)
            weighted_fscores[i].append(metrics["w_f_score"])

    with open("k_eval_data.pickle", "wb") as pf:
        pickle.dump(weighted_fscores, pf)

    ax.boxplot(weighted_fscores, labels=support_sample_sizes)
    plt.savefig("k_eval_boxplot.png")


def evaluate_saved_embeddings(saved_embeddings, saved_labels):
    support_embeddings, support_labels, test_embeddings, test_labels = get_datasets(saved_embeddings, saved_labels)
    return evaluate_dataset(support_embeddings, support_labels, test_embeddings, test_labels)


def evaluate_embeddings(train_embeddings, train_labels, test_embeddings, test_labels, verbose=True):
    num_support_samples = 25
    support_embeddings, support_labels = get_num_of_support_samples(train_embeddings, train_labels, num_support_samples)

    return evaluate_dataset(support_embeddings, support_labels, test_embeddings, test_labels, verbose=verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--single-run", action="store_true",
                        help="Run a single experiment instead of having multiple runs with different amount of "
                             "support embedding.")
    args = parser.parse_args()

    embeddings_filename = 'embeddings_full_ds_baseline.npy'
    labels_filename = 'labels_full_ds_baseline.pickle'
    saved_embeddings = np.load(embeddings_filename)
    with open(labels_filename, 'rb') as f:
        saved_labels = list(pickle.load(f))

    if args.single_run:
        metrics = evaluate_saved_embeddings(saved_embeddings, saved_labels)
        with open("metrics_test.log", "w") as log_file:
            json.dump(metrics, log_file, indent=4)
    else:
        run_experiment(saved_embeddings, saved_labels)


if __name__ == '__main__':
    main()
