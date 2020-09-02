import json

import os
import random
import tarfile

json_paths = ["ganwriting_generated_words_nums_dates_18k.json", "alpha_num_generated_20k.json",
              "gw_baseline_replicated.json", "nums_generated_20k.json", "plzs_generated_20k.json"]
dataset_dir = "../handwriting_embedding/datasets"
tar_dir = "datasets/tars"
new_dataset_name = "full_ds_nums_dates_text_only"

allowed_classes = ["num", "date", "text"]
print(f"Using only camples of classes: {', '.join(allowed_classes)}")

dataset = {}
for dataset_description_path in json_paths:
    with open(os.path.join(dataset_dir, dataset_description_path), 'r') as f:
        json_file = json.load(f)

    for sample in json_file:
        if sample["type"] not in allowed_classes:
            continue
        if sample['type'] not in dataset:
            dataset[sample['type']] = []
        dataset[sample['type']].append(sample)
classes = dataset.keys()

train = []
test = []
smallest_class_len = min([len(ds) for ds in dataset.values()])
threshold = int(smallest_class_len * 0.9)
for samples in dataset.values():
    random.shuffle(samples)  # Makes sure that no internal structure in the json file messes up dataset
    train.extend([sample for sample in samples[:threshold]])
    test.extend([sample for sample in samples[threshold:smallest_class_len]])

random.shuffle(train)
random.shuffle(test)

print(f"Saving {len(train)}/{len(test)} samples")

# make json
train_set_path = os.path.join(dataset_dir, f"{new_dataset_name}_train.json")
with open(train_set_path, "w") as train_out_file:
    json.dump(train, train_out_file, indent=4)

test_set_path = os.path.join(dataset_dir, f"{new_dataset_name}_test.json")
with open(test_set_path, "w") as test_out_file:
    json.dump(test, test_out_file, indent=4)

tar_filename = os.path.join(tar_dir, new_dataset_name + "_train_test.tar.bz2")
image_files = [sample["path"] for sample in test + train]
with tarfile.open(tar_filename, "w:bz2") as tar:
    for filename in image_files:
        tar.add(os.path.join(dataset_dir, filename), arcname=filename)
    tar.add(train_set_path, arcname=os.path.basename(train_set_path))
    tar.add(test_set_path, arcname=os.path.basename(test_set_path))
