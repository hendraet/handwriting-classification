import json

import os
import random
import tarfile

json_paths = ["ganwriting_generated_words_nums_dates_18k.json", "alpha_num_generated_20k.json",
              "gw_baseline_replicated.json", "nums_generated_20k.json", "plzs_generated_20k.json"]
dataset_dir = "../handwriting_embedding/datasets"
tar_dir = "datasets/tars"
new_dataset_name = "5CHPT"
create_validation_set = True

# allowed_classes = ["num", "plz"]
allowed_classes = None
if allowed_classes is not None:
    print(f"Using only samples of classes: {', '.join(allowed_classes)}")

train_set_sample_limit = None
if train_set_sample_limit is not None:
    print(f"Limiting train samples to {train_set_sample_limit} for each class.")

test_set_sample_limit = None
if test_set_sample_limit is not None:
    print(f"Limiting train samples to {test_set_sample_limit} for each class.")

dataset = {}
for dataset_description_path in json_paths:
    with open(os.path.join(dataset_dir, dataset_description_path), "r") as f:
        json_file = json.load(f)

    for sample in json_file:
        if allowed_classes is not None and sample["type"] not in allowed_classes:
            continue

        if sample["type"] not in dataset:
            dataset[sample["type"]] = []
        dataset[sample["type"]].append(sample)
classes = dataset.keys()

train = []
test = []
val = []
max_len_per_class = min([len(ds) for ds in dataset.values()])
if train_set_sample_limit is None:
    if create_validation_set:
        threshold = int(max_len_per_class * 0.8)
    else:
        threshold = int(max_len_per_class * 0.9)
else:
    assert train_set_sample_limit < max_len_per_class
    threshold = train_set_sample_limit

if test_set_sample_limit is not None:
    max_len_per_class = threshold + test_set_sample_limit

second_threshold = threshold + (max_len_per_class - threshold) // 2
for samples in dataset.values():
    random.shuffle(samples)  # Makes sure that no internal structure in the json file messes up dataset
    train.extend([sample for sample in samples[:threshold]])
    if create_validation_set:
        val.extend([sample for sample in samples[threshold:second_threshold]])
        test.extend([sample for sample in samples[second_threshold:max_len_per_class]])
    else:
        test.extend([sample for sample in samples[threshold:max_len_per_class]])


random.shuffle(train)
random.shuffle(test)
random.shuffle(val)

print(f"Saving {len(train)}/{len(val)}/{len(test)} samples")

# make json
train_set_path = os.path.join(dataset_dir, f"{new_dataset_name}_train.json")
with open(train_set_path, "w") as train_out_file:
    json.dump(train, train_out_file, indent=4)

test_set_path = os.path.join(dataset_dir, f"{new_dataset_name}_test.json")
with open(test_set_path, "w") as test_out_file:
    json.dump(test, test_out_file, indent=4)

if create_validation_set:
    val_set_path = os.path.join(dataset_dir, f"{new_dataset_name}_val.json")
    with open(val_set_path, "w") as val_out_file:
        json.dump(val, val_out_file, indent=4)
    tar_filename = os.path.join(tar_dir, new_dataset_name + "_train_val_test.tar.bz2")
else:
    tar_filename = os.path.join(tar_dir, new_dataset_name + "_train_test.tar.bz2")

image_files = [sample["path"] for sample in test + val + train]
with tarfile.open(tar_filename, "w:bz2") as tar:
    for i, filename in enumerate(image_files):
        if (i + 1) % (len(image_files) // 10) == 0:
            print(f"Adding image {i}/{len(image_files)}")
        tar.add(os.path.join(dataset_dir, filename), arcname=filename)
    tar.add(train_set_path, arcname=os.path.basename(train_set_path))
    tar.add(test_set_path, arcname=os.path.basename(test_set_path))
    if create_validation_set:
        tar.add(val_set_path, arcname=os.path.basename(val_set_path))
