import json
import os


def main():
    ds_description_paths = [
        "/home.new/hendrik/datasets/5CHPT_plus_unlabelled/train/5CHPT_plus_unlabelled_train.json",
        "/home.new/hendrik/datasets/5CHPT_plus_unlabelled/test/5CHPT_plus_unlabelled_test.json",
        "/home.new/hendrik/datasets/5CHPT_plus_unlabelled/val/5CHPT_plus_unlabelled_val.json",
        "/home.new/hendrik/datasets/5CHPT_plus_unlabelled/unlabelled/5CHPT_plus_unlabelled_unlabelled.json"
    ]

    for ds_description_path in ds_description_paths:
        with open(ds_description_path) as f:
            samples = json.load(f)

        dirname = os.path.dirname(ds_description_path)
        missing_files = []
        for sample in samples:
            sample_path = os.path.join(dirname, sample["path"])
            if not os.path.exists(sample_path):
                missing_files.append(sample_path)

        if len(missing_files) == 0:
            print(f"The dataset {ds_description_path} is sound.")
        else:
            print("The follwing files are missing")
            print(missing_files)


if __name__ == '__main__':
    main()