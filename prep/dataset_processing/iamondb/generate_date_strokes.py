import argparse
import csv
import itertools
import json
import math
import statistics
from datetime import datetime

import copy
import numpy as np
import os
import random
from PIL import ImageDraw, Image

from prep.image_processing.resize_images import resize_img
from prep.utils import create_tar


def normalise_dataset(dataset):
    for i, stroke_set in enumerate(dataset):
        stroke_coords = stroke_set[:, 1:]

        # relative to absolute
        absolute_stroke_coords = np.cumsum(stroke_coords, axis=0)

        # normalise, so every stroke coordinate is >= 0
        absolute_stroke_coords[:, 0] -= absolute_stroke_coords[:, 0].min()
        absolute_stroke_coords[:, 1] -= absolute_stroke_coords[:, 1].min()

        dataset[i][:, 1] = absolute_stroke_coords[:, 0]
        dataset[i][:, 2] = absolute_stroke_coords[:, 1]

    return dataset


def get_dataset_and_label_infos(args, dataset_prefix):
    dataset_path = os.path.join(args.dataset_dir, "npys", dataset_prefix + ".npy")
    dataset = np.load(dataset_path, allow_pickle=True)
    dataset = normalise_dataset(dataset)

    labels_path = os.path.join(args.description_dir, dataset_prefix + ".csv")
    with open(labels_path, 'r') as label_file:
        reader = csv.reader(label_file)
        label_infos = [row for row in reader]
    return dataset, label_infos


def get_writer_id_for_filename(filename):
    writer_dir = os.path.dirname(filename)
    for blah in ["strokesw.xml", "strokesu.xml", "strokesz.xml", "strokesx.xml", "strokes.xml"]:
        xml_filename = os.path.join("original", writer_dir, blah)
        if os.path.exists(xml_filename):
            break
    assert os.path.exists(xml_filename), f"{xml_filename} does not exist"

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    general = root.find("General")
    if general is None:
        return "no_wid_found"
    wid = general.find("Form").get("writerID")

    return wid


def get_dataset(args, digit_dataset_prefix, dot_dataset_prefix, dash_dataset_prefix):
    digit_dataset, digit_label_infos = get_dataset_and_label_infos(args, digit_dataset_prefix)

    # Map structure looks like this:
    # {
    #     wid: {
    #         char1: [ strokes ],
    #         char2: [ strokes ]
    #     }, ...
    # }
    wid_sample_map = {}
    for label_info, strokes in zip(digit_label_infos, digit_dataset):
        wid = get_writer_id_for_filename(label_info[1])
        label = label_info[0]

        if wid == "no_wid_found":  # TODO: bit magic string
            continue

        if wid not in wid_sample_map:
            wid_sample_map[wid] = {}
        if label not in wid_sample_map[wid]:
            wid_sample_map[wid][label] = []
        wid_sample_map[wid][label].append(strokes)

    dot_dataset, dot_label_infos = get_dataset_and_label_infos(args, dot_dataset_prefix)
    dash_dataset, dash_label_infos = get_dataset_and_label_infos(args, dash_dataset_prefix)
    for i, stroke_set in enumerate(dash_dataset):
        stroke_set[:, 2] = stroke_set[:, 2] + 5
        dash_dataset[i] = stroke_set

    # Other chars are usually only a few samples and are stylistically not as important as numbers. Therefore they are
    # added to the possible strokes of all writers and treated as if they were written by them
    for dataset, label_infos in ((dot_dataset, dot_label_infos), (dash_dataset, dash_label_infos)):
        for v in wid_sample_map.values():
            for label_info, strokes in zip(label_infos, dataset):
                label = label_info[0]
                if label not in v:
                    v[label] = []
                v[label].append(strokes)

    return wid_sample_map


def get_wid_style_map(dataset):
    wid_style_map = {}
    for wid in dataset.keys():
        thickness = random.choice([3, 5, 7])
        stroke_colour = random.randint(0, 40)

        if wid not in wid_style_map:
            wid_style_map[wid] = {}
        wid_style_map[wid]["thickness"] = thickness
        wid_style_map[wid]["stroke_colour"] = (stroke_colour, stroke_colour, stroke_colour)

    return wid_style_map


def generate_date():
    # TODO: '%m/%d/%y', '%m/%d/%Y','%d. %B %Y', '%d. %b %Y', '%B %y', '%d. %B', '%d. %b'
    date_formats = ['%d.%m.%y', '%d.%m.%Y', '%d-%m-%y']

    start = datetime.strptime('01.01.1000', '%d.%m.%Y')
    end = datetime.strptime('01.01.2020', '%d.%m.%Y')
    delta = end - start

    rand_date = start + delta * random.random()

    return rand_date.strftime(random.choice(date_formats))


def get_random_string(string_type):
    if string_type == "date":
        string_type = "date"
        string = generate_date()
    else:
        string_type = "num"
        string = str(random.randint(0, 100000))
        if random.choice([True, False]):
            # Add 0 padding for some nums without increasing max num length
            string = string[:-1].zfill(len(string))

    return string, string_type


def get_stroke_sets_for_string(generated_string, dataset):
    required_chars = set([c for c in generated_string])
    writer_candidates = {wid: v for wid, v in dataset.items() if all(req_char in v.keys() for req_char in required_chars)}
    wid = random.choice(list(writer_candidates.keys()))

    stroke_sets = []
    for char in generated_string:
        random_stroke = random.choice(writer_candidates[wid][char])
        stroke_sets.append(copy.deepcopy(random_stroke))

    return np.asarray(stroke_sets), wid


def create_image_from_strokes(orig_x, orig_y, stroke_ends, style):
    x = copy.deepcopy(orig_x)
    y = copy.deepcopy(orig_y)

    padding = 10
    resize_factor = 3

    y *= -1  # y-axis is inverted
    y -= y.min()

    x *= resize_factor
    y *= resize_factor

    x += padding
    y += padding

    width = math.ceil(x.max() + padding)
    height = math.ceil(y.max() + padding)

    img = Image.new("RGB", (width, height), color=(255, 255, 255))  # TODO width height
    img_canvas = ImageDraw.Draw(img)

    for i, point in enumerate(stroke_ends[:-1]):
        if stroke_ends[i] == 1:
            continue
        line_width = style["thickness"]
        end_x = x[i + 1]
        end_y = y[i + 1]
        colour = style["stroke_colour"]
        img_canvas.line((x[i], y[i], end_x, end_y), fill=colour, width=line_width)
        radius = line_width // 2  # TODO: plus 1 or sth?
        img_canvas.ellipse([end_x - radius, end_y - radius, end_x + radius, end_y + radius], fill=colour, width=0)

    return img


def concatenate_strokes(string, stroke_sets, resize=True):
    heights = [(stroke_set[:, 2].max() - stroke_set[:, 2].min()) for stroke_set in stroke_sets]
    # Exclude punctuation form median calculation to avoid unnecessary distortion
    median_height = statistics.mean([height for i, height in enumerate(heights) if string[i].isdigit()])

    absolute_strokes = []
    for i, stroke_set in enumerate(stroke_sets):
        stroke_coords = stroke_set[:, 1:]

        if string[i].isdigit() and resize:
            distortion_factor = random.uniform(-0.10, 0.10)
            resize_factor = median_height / heights[i] + distortion_factor
            stroke_coords *= resize_factor

        # shift current stroke set, so that it is displayed on the right of the previous stroke set
        if absolute_strokes:
            padding = 5
            stroke_coords[:, 0] += absolute_strokes[-1][:, 0].max() + padding

        absolute_strokes.append(stroke_coords)

    stroke_ends = np.concatenate([el[:, 0] for el in stroke_sets])
    absolute_strokes = np.concatenate(absolute_strokes)

    x = absolute_strokes[:, 0]
    y = absolute_strokes[:, 1]

    return x, y, stroke_ends


def write_synth_results(samples, labels):
    np.save("dates.npy", np.asarray(samples))

    with open("dates.csv", "w") as label_file:
        label_file.write("\n".join(labels) + "\n")


def convert_from_absolute_to_relative(x, y):
    combined = np.stack([x, y], axis=1)
    combined_relative = combined[1:] - combined[:-1]
    combined_relative = np.insert(combined_relative, 0, [0., 0.], axis=0)

    return combined_relative[:, 0], combined_relative[:, 1]


def main():
    string_type_choices = ["num", "date", "all"]

    parser = argparse.ArgumentParser(
        description="A tool for synthesising dates and numbers out of online data.")
    parser.add_argument("dataset_name", help="Name of the dataset that should be generated.")
    parser.add_argument("string_type", choices=string_type_choices,
                        help="The type of string that should be generated. Can be 'num', 'date' or 'all'.")
    parser.add_argument("tar_dir", help="Generated files will be stored as tar in the dir specified here.")
    parser.add_argument("final_dir", help="Dir where the generated files will be moved to.")
    parser.add_argument("--dataset-dir", default="../datasets",
                        help="The directory where the extracted strokes are stored as npy files.")
    parser.add_argument("--description-dir", default="../dataset_descriptions",
                        help="The path of csv file that labels the strokes in the dataset-dir.")
    parser.add_argument("--num-dataset", default="iamondb_num",
                        help="The name of the dataset that contains all the number strokes. This is required for date and "
                             "number generation.")
    parser.add_argument("--dot-dataset", default="iamondb_dot",
                        help="The name of the dataset that contains the dot strokes. This is required for dates only.")
    parser.add_argument("--dash-dataset", default="iamondb_dash",
                        help = "The name of the dataset that contains the dash strokes. This is required for dates only.")
    parser.add_argument("--num", type=int, default="10", help="The number of samples that should be generated.")
    parser.add_argument("--show-image", action="store_true", help="Shows the generated image for each sample.")
    parser.add_argument("--synth-ready", action="store_true",
                        help="Csv and npy files will be generated JSON and images otherwise. Should be used if the "
                             "generated string should be used for further synthesis.")
    args = parser.parse_args()

    out_dir = args.dataset_dir
    image_files = [fn for fn in os.listdir(out_dir) if fn.endswith(".png")]
    assert not image_files, "There are already image files in the out dir"

    if args.string_type != "all":
        print("You are generating a dataset for only one type. Make sure that you re-generate the datasets if you want "
              "to support more types because writer specific style information are not preserved between runs and can "
              "lead to undesired behaviour")
        target_string_types = [args.string_type]
    else:
        target_string_types = [el for el in string_type_choices if el != "all"]

    dataset = get_dataset(args, args.num_dataset, args.dot_dataset, args.dash_dataset)
    wid_style_map = get_wid_style_map(dataset)

    samples = []
    dataset_info = []
    for i, target_string_type in itertools.product(range(0, args.num), target_string_types):
        generated_string, string_type = get_random_string(target_string_type)

        stroke_sets, wid = get_stroke_sets_for_string(generated_string, dataset)

        x, y, stroke_ends = concatenate_strokes(generated_string, stroke_sets, resize=True)
        style = wid_style_map[wid]
        img = create_image_from_strokes(x, y, stroke_ends, style)
        if args.show_image:
            img.show()

        x, y = convert_from_absolute_to_relative(x, y)

        if args.synth_ready:
            new_stroke_set = np.stack((stroke_ends, x, y), axis=1)
            samples.append(new_stroke_set)
            dataset_info.append(generated_string)
        else:
            filename = f"{generated_string}_{wid}.png"
            out_path = os.path.join(out_dir, filename)  # TODO
            resized_img = resize_img(img, (216, 64), padding_color=255)
            with open(out_path, "wb") as out_file:
                resized_img.save(out_file)

            info = {
                "string": generated_string,
                "type": string_type,
                "path": os.path.split(out_path)[1]
            }
            dataset_info.append(info)

    if args.synth_ready:
        write_synth_results(samples, dataset_info)
    else:
        new_dataset_name = args.dataset_name
        new_dataset_description = os.path.join(out_dir, new_dataset_name + ".json")
        with open(new_dataset_description, "w") as out_json:
            json.dump(dataset_info, out_json, ensure_ascii=False, indent=4)

        create_tar(new_dataset_name, new_dataset_description, out_dir, args.tar_dir, args.final_dir)


if __name__ == '__main__':
    main()
