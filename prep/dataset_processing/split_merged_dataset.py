import json

import os


def main():
    new_ds_name = "5CHPT_unlabelled.json"
    all_samples_description_path = "/home.new/hendrik/datasets/5CHPT_plus_unlabelled/train_plus_unlabelled/5CHPT_plus_unlabelled_train.json"
    split_ds_description_path = "/home.new/hendrik/datasets/5CHPT_plus_unlabelled/train/5CHPT_train.json"
    all_samples_base_dir = os.path.dirname(all_samples_description_path)

    with open(all_samples_description_path) as all_samples_description_file:
        all_samples = json.load(all_samples_description_file)

    with open(split_ds_description_path) as split_ds_description_file:
        samples_to_be_removed = json.load(split_ds_description_file)

    num_all_samples = len(all_samples)
    print(f"Num all samples: {num_all_samples}")
    num_samples_to_be_removed = len(samples_to_be_removed)
    print(f"Num smaples to be removed: {num_samples_to_be_removed}")

    for sample in samples_to_be_removed:
        all_samples.remove(sample)
        sample_filename = os.path.join(all_samples_base_dir, sample["path"])
        if os.path.exists(sample_filename):
            os.remove(sample_filename)

    assert len(all_samples) == num_all_samples - num_samples_to_be_removed

    for sample in all_samples:
        sample_filename = os.path.join(all_samples_base_dir, sample["path"])
        if not os.path.exists(sample_filename):
            print(f"{sample['path']} is missing")

    with open(os.path.join(all_samples_base_dir, new_ds_name), "w") as out_f:
        json.dump(all_samples, out_f, indent=4, ensure_ascii=False)

    print("Done")


if __name__ == '__main__':
    main()
