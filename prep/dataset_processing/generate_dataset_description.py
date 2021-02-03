import argparse
import json

import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="directory that contains the images and the description")
    parser.add_argument("description_filename", type=str, help="name of the description file that will be created")
    parser.add_argument("--omit-label", action="store_true", default=False,
                        help="if set, the string field in the resulting description will be empty")
    parser.add_argument("--custom-type", type=str,
                        help="Don't deduce type based on the string but use this custom type")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    description_filename = args.description_filename
    description_path = os.path.join(dataset_dir, description_filename + ".json")
    custom_type = args.custom_type

    description = []
    for path in os.listdir(dataset_dir):
        filename, ext = os.path.splitext(path)
        if ext != ".png":
            continue

        string = ""
        if not args.omit_label:
            string = filename.split("_")[0]

        if custom_type is not None:
            string_type = custom_type
        else:
            # TODO: this check is not perfect
            if all(char.isdigit() for char in string):
                string_type = "num"
            elif all(char.isalpha() for char in string):
                string_type = "text"
            else:
                string_type = "date"

        entry = {
            "string": string,
            "type": string_type,
            "path": path
        }
        description.append(entry)

    with open(description_path, "w") as out_json:
        json.dump(description, out_json, indent=4)


if __name__ == '__main__':
    main()
