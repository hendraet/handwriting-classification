import numpy as np

in_path_ds1 = "../datasets/dates.npy"
in_path_labels1 = "../dataset_descriptions/dates.csv"
in_path_ds2 = "../../../Handwriting-synthesis/data/strokes.npy"
in_path_labels2 = "../../../Handwriting-synthesis/data/sentences.txt"
out_path_ds = "../../../Handwriting-synthesis/data/sentences_dates.npy"
out_path_labels = "../../../Handwriting-synthesis/data/sentences_dates.txt"

ds1 = np.load(in_path_ds1, allow_pickle=True, encoding="bytes")
ds2 = np.load(in_path_ds2, allow_pickle=True, encoding="bytes")

with open(in_path_labels1) as label_file1:
    labels1 = np.asarray(label_file1.readlines())

with open(in_path_labels2) as label_file2:
    labels2 = np.asarray(label_file2.readlines())

tmp1 = np.stack((labels1[:-1], ds1), axis=1)
tmp2 = np.stack((labels2[:-1], ds2), axis=1)

combined = np.concatenate((tmp1, tmp2))
np.random.shuffle(combined)

np.save(out_path_ds, combined[:, 1], allow_pickle=True)

with open(out_path_labels, "w") as out:
    strings = combined[:, 0]
    for i, string in enumerate(strings):
        out.write(string)
