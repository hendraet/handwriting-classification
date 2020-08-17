import json
import os
import shutil

# GANWriting
# dataset_dir = "/home/hendrik/pycharm_upload/handwriting_embedding/datasets"
dataset_dir = "/home/hendrik/research-GANwriting/eval_files/mixed_imgs"
description_filename = "ganwriting_generated_words_nums_dates_18k"
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

# WPI
# wpi_dir = "../orig_datasets/WPI"
# dataset_dir = "../orig_datasets/WPI/labeled_dataset"
# description_filename = "wpi_words_dates_nums_alphanum"
# description_path = os.path.join(dataset_dir, description_filename + ".json")
#
# subdirs = [path for path in os.listdir(wpi_dir) if os.path.isdir(os.path.join(wpi_dir, path)) and path != dataset_dir]
#
# filenames = []
# dataset_description = []
# for subdir in subdirs:
#     full_sub_dir_path = os.path.join(wpi_dir, subdir)
#     relevant_files = [f for f in os.listdir(full_sub_dir_path) if "#" in f and os.path.isfile(os.path.join(full_sub_dir_path, f))]
#     for filename in relevant_files:
#         label, metadata = filename.split("#")
#
#         correct_transcription = True
#         if label[0] == "?":
#             label = label[1:]
#             correct_transcription = False
#
#         cleansed_label = label.replace("_", " ")
#
#         metadata = os.path.splitext(metadata.split("_")[0])
#         string_type = metadata[0]
#
#         new_filename = f"{subdir}_{label}"
#         if new_filename in filenames:
#             unique_filename_found = False
#             running_suffix = 1
#             while not unique_filename_found:
#                 new_filename_w_suffix = f"{new_filename}_{str(running_suffix)}"
#                 if not new_filename_w_suffix in filenames:
#                     unique_filename_found = True
#                     new_filename = new_filename_w_suffix
#                 running_suffix += 1
#
#         new_path = new_filename + ".png"
#
#         infos = {
#             "correct_transcription": correct_transcription,
#             "string": cleansed_label,
#             "type": string_type,
#             "path": new_path
#         }
#         dataset_description.append(infos)
#         filenames.append(new_filename)
#
#         # TODO: cp all the files to dataset dir
#         shutil.copy(os.path.join(full_sub_dir_path, filename), os.path.join(dataset_dir, new_path))
#
# with open(description_path, "w") as out_json:
#     json.dump(dataset_description, out_json, indent=4)
#
