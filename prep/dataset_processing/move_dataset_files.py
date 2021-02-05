import json
import shutil

import os


def main():
    dataset_descriptions = [
        "../handwriting_embedding/datasets/5CHPT_train.json",
        # "../handwriting_embedding/datasets/5CHPT_test.json",
        # "../handwriting_embedding/datasets/5CHPT_val.json"
    ]
    dest_dir = "/home.new/hendrik/datasets/5CHPT_plus_unlabelled"

    for desc in dataset_descriptions:
        with open(desc) as desc_f:
            json_contents = json.load(desc_f)

        dataset_dir = os.path.dirname(desc)

        new_dir_name = os.path.splitext(os.path.basename(desc))[0].split("_")[1]
        full_path_new_dir = os.path.join(dest_dir, new_dir_name)
        os.makedirs(full_path_new_dir)
        for sample in json_contents:
            sample_path = os.path.join(dataset_dir, sample["path"])
            shutil.copy(sample_path, full_path_new_dir)
        shutil.copy(desc, full_path_new_dir)


if __name__ == '__main__':
    main()
