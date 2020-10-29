import argparse
import json

import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    # should be root folder for all the original folders when not rebuilding
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("description_filename", type=str)
    parser.add_argument("-r", "--rebuild", type=str,
                        help="Will rebuild dataset based on give dataset description and present files")
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    description_filename = args.description_filename
    description_path = os.path.join(out_dir, description_filename + ".json")

    if args.rebuild is not None:
        original_description_filename = args.rebuild
        with open(original_description_filename) as json_file:
            json_contents = json.load(json_file)

        new_description = []
        for sample in json_contents:
            full_path = os.path.join(out_dir, sample["path"])
            if os.path.exists(full_path):
                new_description.append(sample)

        with open(os.path.join(out_dir, description_filename), "w") as out_f:
            json.dump(new_description, out_f, indent=4)

    else:
        subdirs = [path for path in os.listdir(in_dir) if
                   os.path.isdir(os.path.join(in_dir, path)) and path != out_dir]

        filenames = []
        dataset_description = []
        for subdir in subdirs:
            full_sub_dir_path = os.path.join(in_dir, subdir)
            relevant_files = [f for f in os.listdir(full_sub_dir_path) if
                              "#" in f and os.path.isfile(os.path.join(full_sub_dir_path, f))]
            for filename in relevant_files:
                label, metadata = filename.split("#")

                correct_transcription = True
                if label[0] == "?":
                    label = label[1:]
                    correct_transcription = False

                cleansed_label = label.replace("_", " ")

                metadata = os.path.splitext(metadata.split("_")[0])
                string_type = metadata[0]

                new_filename = f"{subdir}_{label}"
                if new_filename in filenames:
                    unique_filename_found = False
                    running_suffix = 1
                    while not unique_filename_found:
                        new_filename_w_suffix = f"{new_filename}_{str(running_suffix)}"
                        if not new_filename_w_suffix in filenames:
                            unique_filename_found = True
                            new_filename = new_filename_w_suffix
                        running_suffix += 1

                new_path = new_filename + ".png"

                infos = {
                    "correct_transcription": correct_transcription,
                    "string": cleansed_label,
                    "type": string_type,
                    "path": new_path
                }
                dataset_description.append(infos)
                filenames.append(new_filename)

                # TODO: cp all the files to dataset dir
                shutil.copy(os.path.join(full_sub_dir_path, filename), os.path.join(out_dir, new_path))

        with open(description_path, "w") as out_json:
            json.dump(dataset_description, out_json, indent=4)



if __name__ == '__main__':
    main()
