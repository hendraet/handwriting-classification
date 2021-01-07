import argparse
import json

import os
import random
import tarfile


def get_args():
    parser = argparse.ArgumentParser(description="Merging multiple datasets and generating new descriptions and tars")
    parser.add_argument("new_dataset_name", type=str, help="the name of the new dataset")
    parser.add_argument("dataset_dir", type=str, help="the directory where the dataset descriptions are located")
    parser.add_argument("tar_dir", type=str, help="the directory where the resulting tar archive should be stored")
    parser.add_argument("json_paths", type=str, nargs="+",
                        help="the filename of all dataset descriptions (json) that should be merged")
    parser.add_argument("-ac", "--allowed-classes", type=str, nargs="+",
                        help="only use the given string types for the new dataset")
    parser.add_argument("-pp", "--partition-percentages", type=float, nargs="+", default=[0.9, 0.1],
                        help="list of percentages that define how big each partition should be (have to sum up to 1.0)")
    parser.add_argument("-ps", "--partition-names", type=str, nargs="+", default=["train", "test"],
                        help="names of the partitions, e.g. 'train' and 'test'")
    parser.add_argument("-sl", "--sample-limits", type=int,
                        help="limits the number of samples for each partition, 'None' if a partition shouldn't be "
                             "limited")
    args = parser.parse_args()

    if args.sample_limits is None:
        args.sample_limits = [None] * len(args.partition_percentages)

    assert sum(args.partition_percentages) == 1.0
    assert len(args.partition_percentages) == len(args.sample_limits)
    assert len(args.partition_percentages) == len(args.partition_names)

    return args


def get_new_dataset_partitions(json_paths, dataset_dir, balance_classes=True, partition_percentages=(1.0,),
                               sample_limits=(None,), allowed_classes=None):
    if not balance_classes:
        raise NotImplementedError

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

    max_len_per_class = min([len(ds) for ds in dataset.values()])  # Make dataset balanced
    partition_indices = []
    for i, partition_percentage in enumerate(partition_percentages):
        partition_start_idx = int(max_len_per_class * sum(partition_percentages[:i]))
        if sample_limits[i] is None:
            partition_end_idx = partition_start_idx + int(max_len_per_class * partition_percentage)
        else:
            assert sample_limits[i] < max_len_per_class
            partition_end_idx = partition_start_idx + sample_limits[i]
        partition_indices.append((partition_start_idx, partition_end_idx))

    new_dataset_partitions = [[] for _ in partition_indices]
    for samples in dataset.values():
        random.shuffle(samples)  # Makes sure that no internal structure in the json file messes up dataset
        for i, (start_idx, end_idx) in enumerate(partition_indices):
            new_partition = [sample for sample in samples[start_idx:end_idx]]
            new_dataset_partitions[i].extend(new_partition)

    for partition in new_dataset_partitions:
        random.shuffle(partition)

    return tuple(new_dataset_partitions)


def main():
    args = get_args()
    if args.allowed_classes is not None:
        print(f"Using only samples of classes: {', '.join(args.allowed_classes)}")

    info_str = ", ".join([f"{name}: {str(limit) if limit is not None else '-'}" for name, limit in
                          zip(args.partition_names, args.sample_limits)])
    print(f"Limiting sample numbers to {info_str}")

    partitions = get_new_dataset_partitions(args.json_paths, args.dataset_dir,
                                            partition_percentages=args.partition_percentages,
                                            sample_limits=args.sample_limits,
                                            allowed_classes=args.allowed_classes)

    print(f"Saving {'/'.join([str(len(partition)) for partition in partitions])} samples")

    tar_filename = os.path.join(args.tar_dir, f"{args.new_dataset_name}_{'_'.join(args.partition_names)}.tar.bz2")
    # creating an extra set for the image paths to get rid of duplicates
    image_paths = set()
    with tarfile.open(tar_filename, "w:bz2") as tar:
        for partition_suffix, partition in zip(args.partition_names, partitions):
            path = os.path.join(args.dataset_dir, f"{args.new_dataset_name}_{partition_suffix}.json")
            with open(path, "w") as out_file:
                json.dump(partition, out_file, indent=4)

            for i, sample in enumerate(partition):
                image_paths.add(sample["path"])
            tar.add(path, arcname=os.path.basename(path))

        num_samples = len(image_paths)
        for path in image_paths:
            if num_samples // 10 > 0 and (i + 1) % (num_samples // 10) == 0:  # avoid mod 0 if dataset too small
                print(f"Adding image {i}/{num_samples}")
            tar.add(os.path.join(args.dataset_dir, path), arcname=path)


if __name__ == '__main__':
    main()
