import pickle

import matplotlib
import numpy as np
from scipy.cluster.vq import kmeans2
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


def classify_embeddings(embeddings, support_labels, support_embeddings, actual_labels=None):
    # TODO: base num centroids on num classes for outlier detection
    num_centroids = len(set(support_labels))
    centroid, labels = kmeans2(embeddings, num_centroids, minit='points')

    # labeling the centroids with knn on test embeddings
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(support_embeddings, support_labels)
    predicted_centroid_labels = knn.predict(centroid)

    if actual_labels is not None:
        translated_labels = translate_labels(actual_labels, labels, predicted_centroid_labels)
        draw_embeddings_cluster('comp_num_vs_text.png', embeddings, translated_labels, centroid)

    return [predicted_centroid_labels[label] for label in labels]


def main():
    # TODO: make possible to only pass model and support images
    saved_embeddings = np.load('embeddings.npy')
    with open('labels.pickle', 'rb') as f:
        saved_labels = list(pickle.load(f))

    # Mock more data
    # saved_embeddings = np.append(saved_embeddings, [(saved_embeddings[0] + saved_embeddings[1]) / 2], axis=0)
    # saved_embeddings = np.append(saved_embeddings, [(saved_embeddings[1] + saved_embeddings[2]) / 2], axis=0)
    # saved_labels.append('blah')
    # saved_labels.append('blah')

    indices = [0, 1, 2, 3, 4, 5, 6, 8, 11, 14]
    support_labels = np.take(saved_labels, indices)
    support_embeddings = np.take(saved_embeddings, indices, axis=0)

    # arguable if test dataset can be part of the embeddings or not
    saved_labels = np.delete(saved_labels, indices, axis=0)
    saved_embeddings = np.delete(saved_embeddings, indices, axis=0)

    predicted_labels = classify_embeddings(saved_embeddings, support_labels, support_embeddings, saved_labels)
    # TODO: get all labels where predicted label differs from actual label and plot them

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
