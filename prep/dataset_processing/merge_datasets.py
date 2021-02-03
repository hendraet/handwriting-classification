import argparse
import json
import textwrap

import shutil

import os
import random
import tarfile


def get_args():
    help_str = '''
    Merging multiple datasets and generating new descriptions and tars 
    
    The script creates the follwing dataset file structure:
    - {dataset_dir}
        - dataset root: {new_dataset_name}, e.g. "cifar10"
            - partition0: {partition_names[0]}, e.g. "train"
                - dataset description {new_dataset_name}_{partition_names[0]}.json, e.g. "cifar10_train.json"
                - [images that are part of this partition]
            - [more partitions]
                - ...
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(help_str))
    parser.add_argument("new_dataset_name", type=str, help="the name of the new dataset")
    parser.add_argument("dataset_dir", type=str, help="the directory where the resulting dataset should be saved to")
    parser.add_argument("tar_dir", type=str, help="the directory where the resulting tar archive should be stored")
    parser.add_argument("json_paths", type=str, nargs="+",
                        help="the paths of all dataset descriptions (json) that should be merged")
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


def get_new_dataset_partitions(json_paths, balance_classes=True, partition_percentages=(1.0,), sample_limits=(None,),
                               allowed_classes=None):
    if not balance_classes:
        raise NotImplementedError

    dataset = {}
    for dataset_description_path in json_paths:
        with open(dataset_description_path, "r") as f:
            json_file = json.load(f)

        for sample in json_file:
            if allowed_classes is not None and sample["type"] not in allowed_classes:
                continue

            if sample["type"] not in dataset:
                dataset[sample["type"]] = []
            dataset[sample["type"]].append({
                "sample_info": sample,
                "orig_dir": os.path.dirname(dataset_description_path)
            })

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

    partitions = get_new_dataset_partitions(args.json_paths,
                                            partition_percentages=args.partition_percentages,
                                            sample_limits=args.sample_limits,
                                            allowed_classes=args.allowed_classes)

    print(f"Saving {'/'.join([str(len(partition)) for partition in partitions])} samples")

    new_dataset_dir = os.path.join(args.dataset_dir, args.new_dataset_name)
    os.makedirs(new_dataset_dir)
    tar_filename = os.path.join(args.tar_dir, f"{args.new_dataset_name}_{'_'.join(args.partition_names)}.tar.bz2")
    # creating an extra set for the image paths to get rid of duplicates
    with tarfile.open(tar_filename, "w:bz2") as tar:
        for partition_suffix, partition in zip(args.partition_names, partitions):
            image_paths = {}
            partition_dir = os.path.join(new_dataset_dir, partition_suffix)
            os.makedirs(partition_dir)

            out_json_path = os.path.join(partition_dir, f"{args.new_dataset_name}_{partition_suffix}.json")
            all_sample_info = [sample["sample_info"] for sample in partition]
            with open(out_json_path, "w") as out_file:
                json.dump(all_sample_info, out_file, indent=4)

            for i, sample in enumerate(partition):
                filename = sample["sample_info"]["path"]
                orig_dir = sample["orig_dir"]
                if filename in image_paths:
                    if orig_dir != image_paths[filename]:
                        print(f"A file with the name {filename} is part of at least two datasets: "
                              f"{os.path.basename(image_paths[filename])}, {os.path.basename(orig_dir)}")
                    else:
                        print(f"A file with the name {filename} is contained multiple times in the dataset "
                              f"{os.path.basename(image_paths[filename])}")
                else:
                    image_paths[filename] = orig_dir
                    src_path = os.path.join(orig_dir, filename)
                    dest_path = os.path.join(partition_dir, filename)
                    shutil.copy(src_path, dest_path)

            tar.add(partition_dir, arcname=os.path.basename(partition_dir))


if __name__ == '__main__':
    main()
