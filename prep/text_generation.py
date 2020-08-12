import argparse
import json
import string
from datetime import datetime

import os
import random
from PIL import Image, ImageDraw, ImageFont
from nltk.corpus import words

from prep.resize_images import resize_img, create_tar


def generate_number():
    if random.choice((True, False)):
        return str(random.randint(1000, 1000000))
    else:
        return str(random.randint(0, 100))


def generate_date():
    # Would be nice to support:
    # DD.MM.YY
    # DD.MM.YYYY
    # MM/DD/YYYY
    # MM/DD/YY
    # DD. Mon YYYY
    # DD. Month YYYY
    # Month YY
    # DD. Month
    # DD. Mon
    date_formats = ["%d.%m.%y", "%d.%m.%Y", "%m/%d/%y", "%m/%d/%Y", "%d. %B %Y", "%d. %b %Y", "%B %y", "%d. %B",
                    "%d. %b", "%d-%m-%y"]

    start = datetime.strptime("01.01.1000", "%d.%m.%Y")
    end = datetime.strptime("01.01.2020", "%d.%m.%Y")
    delta = end - start

    rand_date = start + delta * random.random()

    return rand_date.strftime(random.choice(date_formats))


def generate_text(word_list):
    return random.choice(word_list)


def generate_alpha_num():
    # TODO: make sure at least one num and one char
    length = random.randint(3, 12)
    possible_chars = string.ascii_uppercase + string.ascii_lowercase + string.digits + "_"
    x = ''.join(random.choice(possible_chars) for _ in range(length))
    print(x)
    return x


def generate_plz():
    return str(random.randint(0, 100000)).zfill(5)


def generate_image(txt, add_additional_padding=False):
    fontsize = 70
    font_dir = "../google-fonts"
    font_name = random.choice(os.listdir(font_dir))
    # font_name = "Dhurjati-Regular.ttf"
    font = font_dir + "/" + font_name
    target_dimensions = (216, 64)

    img_font = ImageFont.truetype(font, fontsize)
    text_dimensions = img_font.getsize(txt)

    img = Image.new("L", text_dimensions, 255)
    d = ImageDraw.Draw(img)

    d.text((0, 0), txt, fill=0, font=img_font)

    if add_additional_padding:
        padding_right = max(0, int(random.gauss(0, 35)))
        padded_img = resize_img(img, target_dimensions, padding_color=255, additional_padding=(0, 0, padding_right, 0))
    else:
        padded_img = resize_img(img, target_dimensions, padding_color=255)

    return padded_img, font_name, text_dimensions


def main():
    parser = argparse.ArgumentParser(description="Generating images with random text")
    parser.add_argument("-d", "--intermediate-dir", type=str, default="datasets/",
                        help="path to directory where image will be stored temporarily. This dir has to be free of "
                             "images")
    parser.add_argument("-td", "--tar-dir", type=str, default="datasets/tars",
                        help="path to directory where the dataset tar should be stored")
    parser.add_argument("-fd", "--final-dir", type=str,
                        help="path to directory where the model will read the data from")
    parser.add_argument("-n", "--num", type=int, default=1, help="num of images to be generated")
    parser.add_argument("-t", "--type", type=str, choices=["date", "text", "num", "plz", "alpha_num"], default="date",
                        help="type of text to be generated or replicate strings from json")
    parser.add_argument("-s", "--show", action="store_true", help="show image(s) after generation")
    parser.add_argument("-p", "--json_img_path", type=str, default="",
                        help="dir path that should be prepended to every path in json file")
    parser.add_argument("--save", action="store_true", help="save image(s) after generation")
    parser.add_argument("-dn", "--dataset-name", type=str, help="name of the new dataset")
    parser.add_argument("-ji", "--in-json-path", type=str,
                        help="name of the json file that contains strings that should be replicated")
    args = parser.parse_args()

    string_type = args.type
    show_image = args.show
    save_image = args.save
    intermediate_dir = args.intermediate_dir
    json_img_dir = args.json_img_path
    out_json_path = intermediate_dir + args.dataset_name + ".json"
    img_list = []

    if string_type == "text":
        # installation of word list is needed on first installation of nltk.
        # nltk.download("words")
        max_length = 9
        word_list = words.words()
        word_list = [word for word in word_list if len(word) < max_length]

    dataset = []
    if args.in_json_path is not None:
        with open(args.in_json_path, "r") as in_json:
            descr = json.load(in_json)
        for entry in descr:
            dataset.append((entry["string"], entry["type"]))
    else:
        for i in range(0, args.num):
            if string_type == "date":
                dataset.append((generate_date(), string_type))
            elif string_type == "text":
                dataset.append((generate_text(word_list), string_type))
            elif string_type == "num":
                dataset.append((generate_number(), string_type))
            elif string_type == "plz":
                dataset.append((generate_plz(), string_type))
            elif string_type == "alpha_num":
                dataset.append((generate_alpha_num(), string_type))
            else:
                print("Unexpected type")
                exit(1)

    if save_image:
        image_files = [fn for fn in os.listdir(intermediate_dir) if fn.endswith(".png")]
        assert not image_files, "There are already image files in the out dir"

    for generated_string, string_type in dataset:
        img, font_name, text_dimensions = generate_image(generated_string)
        cleansed_string = generated_string.replace("/", "_").replace(" ", "_")
        img_name = cleansed_string + "-" + os.path.splitext(font_name)[0]
        img_path = intermediate_dir + img_name + ".png"
        json_img_path = json_img_dir + img_name + ".png"

        if show_image:
            img.show()

        if save_image:
            with open(img_path, "wb+") as img_file:
                img.save(img_file, format="PNG")

        info = {
            "string": generated_string,
            "type": string_type,
            "path": json_img_path,
            "font": font_name
        }
        img_list.append(info)

    if save_image:
        with open(out_json_path, "w+") as json_file:
            json.dump(img_list, json_file, indent=4)

    create_tar(args.dataset_name, out_json_path, intermediate_dir, args.tar_dir, args.final_dir)


if __name__ == "__main__":
    # assert False, "Dates DD Month YYYY have to be fixed"
    main()
