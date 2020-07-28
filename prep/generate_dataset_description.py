import json

import os

dataset_dir = "/home/hendrik/pycharm_upload/handwriting-embedding/datasets"
description_filename = "ganwriting_generated_words_dates_20k"
description_path = os.path.join(dataset_dir, description_filename + ".json")

description = []
for path in os.listdir(dataset_dir):
    filename, ext = os.path.splitext(path)
    if ext != ".png":
        continue

    string = filename.split("_")[0]
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
