import json
from argparse import ArgumentParser

import os
from dateutil.parser import parse

from prep.image_processing.resize_images import create_tar


def get_class_for_label(label):
    # choices = ["date", "text", "num", "plz", "alpha_num", "spec_char", "shape"],
    if str(label).isdigit():
        cls = "num"  # TODO: could be plz as well
    # since the RIMES dataset contains French words a lot of words containing an apostrophe, e.g.
    # "j'ai" are labelled as "rest".
    elif str(label).isalpha() or label.replace("'", "").isalpha():
        cls = "text"
    elif str(label).isalnum():
        cls = "alpha_num"
    else:
        # Naive way to check if string can be interpreted as date
        try:
            parse(label, fuzzy=False)
            cls = "date"
        except (ValueError, TypeError):
            # TODO: some French words that contain a dash, such as "fait-il" are still incorrectly labelled
            #  as "rest"
            cls = "rest"

    return cls


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", default="../orig_datasets/rimes_flat",
                        help="location of the original RIMES dataset")
    parser.add_argument("--labelled", action="store_true", default=False,
                        help="If used the resulting JSON file will contain the type for each sample. Otherwise this "
                             "field will be empty.")
    parser.add_argument("-dn", "--dataset-name", type=str, help="name of the new dataset")
    parser.add_argument("-td", "--tar-dir", type=str, default="datasets/tars",
                        help="path to directory where the dataset tar should be stored")
    parser.add_argument("-fd", "--final-dir", type=str,
                        help="path to directory where the results should be moved to")
    args = parser.parse_args()

    assert os.path.exists(args.tar_dir), "tar directory does not exist"
    if args.final_dir is not None:
        assert os.path.exists(args.final_dir), "Final directory does not exist"

    dataset_dir = args.dataset_dir
    labelled = args.labelled
    ground_truth_files = [
        "ground_truth_training_icdar2011.txt",
        "ground_truth_validation_icdar2011.txt",
        "ground_truth_test_icdar2011.txt",
    ]

    # The data structure might seem a bit odd for the task the script performs, but this way it can be easily extended
    categorised_samples = [{} for i in range(len(ground_truth_files))]
    for i, gt_filename in enumerate(ground_truth_files):
        full_path = os.path.join(dataset_dir, gt_filename)
        with open(full_path, "r") as gt_file:
            lines = gt_file.readlines()

        for line in lines:
            line = line.rstrip()
            if line:
                path, label = line.split()
                string_cls = get_class_for_label(label) if labelled else ""

                if string_cls not in categorised_samples[i]:
                    categorised_samples[i][string_cls] = []
                base_path = os.path.basename(path)
                assert os.path.exists(os.path.join(dataset_dir, base_path)), "This script expects a flat hierachy, i.e all images have to be in the root directory. This can be done by executing the following command in the root dir of the original RIMES dataset: find . -mindepth 2 -type f -exec mv -t . -i '{}' +"
                categorised_samples[i][string_cls].append({
                    "string": label,
                    "type": string_cls,
                    "path": base_path
                })

    sample_infos = []
    # TODO: add method that retains train/val/test split
    for subset in categorised_samples:
        for str_type, samples in subset.items():
            sample_infos.extend(samples)

    if args.dataset_name is None:
        dataset_name = f"rimes_{'labelled' if labelled else 'unlabelled'}"
    else:
        dataset_name = args.dataset_name

    out_json_path = f"{os.path.join(dataset_dir, dataset_name)}.json"
    with open(out_json_path, "w+") as json_file:
        json.dump(sample_infos, json_file, ensure_ascii=False, indent=4)

    create_tar(dataset_name, out_json_path, dataset_dir, args.tar_dir, move_images_instead_of_copy=False,
               final_dir=args.final_dir)
    os.remove(out_json_path)


if __name__ == '__main__':
    main()
