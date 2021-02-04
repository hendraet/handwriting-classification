import argparse
import json
import string
from datetime import datetime

import nltk
import os
import random
from PIL import Image, ImageDraw, ImageFont
from nltk.corpus import words
from skimage.draw import random_shapes

from prep.image_processing.resize_images import resize_img
from prep.utils import create_tar


def generate_number():
    if random.choice((True, False)):
        return str(random.randint(1000, 1000000))
    else:
        return str(random.randint(0, 100))


def generate_date():
    # TODO: better dates (including month words especially)
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
    x = "".join(random.choice(possible_chars) for _ in range(length))
    return x


def generate_plz():
    return str(random.randint(0, 100000)).zfill(5)


def generate_special_chars():
    length = random.randint(3, 12)
    possible_chars = string.punctuation + len(string.punctuation) * " "
    x = "".join(random.choice(possible_chars) for _ in range(length))
    return x


def get_string_dataset(args, string_type):
    if string_type == "text":
        max_length = 9
        try:
            word_list = words.words()
        except LookupError:
            nltk.download("words")
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
                generated_string = generate_date()
            elif string_type == "text":
                generated_string = generate_text(word_list)
            elif string_type == "num":
                generated_string = generate_number()
            elif string_type == "plz":
                generated_string = generate_plz()
            elif string_type == "alpha_num":
                generated_string = generate_alpha_num()
            elif string_type == "spec_char":
                dataset.append((generate_special_chars(), string_type))
                generated_string = generate_special_chars()
            elif string_type == "shape":
                generated_string = ""
            else:
                print("Unexpected type")
                exit(1)
            dataset.append((generated_string, string_type))

    return dataset


def generate_image(txt, font_dir, img_size, add_additional_padding=False):
    fontsize = 70
    font_name = random.choice(os.listdir(font_dir))
    font = font_dir + "/" + font_name

    img_font = ImageFont.truetype(font, fontsize)
    text_dimensions = img_font.getsize(txt)

    img = Image.new("L", text_dimensions, 255)
    d = ImageDraw.Draw(img)

    d.text((0, 0), txt, fill=0, font=img_font)

    if add_additional_padding:
        padding_right = max(0, int(random.gauss(0, 35)))
        padded_img = resize_img(img, img_size, padding_color=255, additional_padding=(0, 0, padding_right, 0))
    else:
        padded_img = resize_img(img, img_size, padding_color=255)

    return padded_img, font_name, text_dimensions


def generate_shapes(img_size):
    img_array, labels = random_shapes(img_size, min_shapes=2, max_shapes=10, multichannel=False, allow_overlap=True)
    img = Image.fromarray(img_array)
    return img


def generate_images_for_string_type(string_type, args):
    dataset = get_string_dataset(args, string_type)

    img_list = []
    for i, (generated_string, string_type) in enumerate(dataset):
        # Generate image
        if string_type == "shape":
            img = generate_shapes(args.image_size)
            font_name = ""
        else:
            img, font_name, _ = generate_image(generated_string, args.font_dir, args.pil_image_size)

        # Generate filename
        if string_type in ["spec_char", "shape"]:
            max_num_id_digits = len(str(args.num))
            cleansed_string = f"{i:0{max_num_id_digits}d}_{string_type}"
        else:
            cleansed_string = generated_string.replace("/", "_").replace(" ", "_")
        img_name = f"generated_{string_type}_{cleansed_string}"
        if font_name != "":  # shapes don't have a font
            img_name += f"-{os.path.splitext(font_name)[0]}"

        img_path = os.path.join(args.intermediate_dir, f"{img_name}.png")
        json_img_path = os.path.join(args.json_img_dir, f"{img_name}.png")

        if args.show_images:
            img.show()

        if args.save_images:
            with open(img_path, "wb+") as img_file:
                img.save(img_file, format="PNG")

        info = {
            "string": generated_string,
            "type": string_type if not args.unlabelled else "",
            "path": json_img_path,
            "font": font_name
        }
        img_list.append(info)

    return img_list


def main():
    parser = argparse.ArgumentParser(description="Generating images with random text")
    parser.add_argument("types", type=str, nargs="+",
                        choices=["date", "text", "num", "plz", "alpha_num", "spec_char", "shape"],
                        help="type of text to be generated")
    parser.add_argument("-d", "--intermediate-dir", type=str, default="datasets/",
                        help="path to directory where image will be stored temporarily. This dir has to be free of "
                             "images")
    parser.add_argument("-td", "--tar-dir", type=str, default="datasets/tars",
                        help="path to directory where the dataset tar should be stored")
    parser.add_argument("-fd", "--final-dir", type=str,
                        help="path to directory where the results should be moved to")
    parser.add_argument("-fod", "--font-dir", type=str, default="../google-fonts",
                        help="path to directory where the font data lies")
    parser.add_argument("-n", "--num", type=int, default=1, help="num of images to be generated")
    parser.add_argument("-s", "--show-images", action="store_true", help="show image(s) after generation")
    parser.add_argument("-p", "--json-img-dir", type=str, default="",
                        help="dir path that should be prepended to every path in json file")
    parser.add_argument("--save-images", action="store_true", help="save image(s) after generation")
    parser.add_argument("-dn", "--dataset-name", type=str, help="name of the new dataset")
    parser.add_argument("-ji", "--in-json-path", type=str,
                        help="name of the json file that contains strings that should be replicated")
    parser.add_argument("-is", "--image-size", type=int, nargs=2, default=[64, 216],
                        help="size of the resulting images in the format (height, width)")
    parser.add_argument("--unlabelled", action="store_true", default=False,
                        help="if set no type label is assigned to the samples")
    args = parser.parse_args()

    if args.save_images:
        image_files = [fn for fn in os.listdir(args.intermediate_dir) if fn.endswith(".png")]
        assert not image_files, "There are already image files in the out dir"

    types = args.types
    out_json_path = os.path.join(args.intermediate_dir, args.dataset_name + ".json")
    args.pil_image_size = (args.image_size[1], args.image_size[0])
    print(f"Generating images of size {args.image_size[0]}x{args.image_size[1]} (height x width)")

    full_img_list = []
    for string_type in types:
        img_list = generate_images_for_string_type(string_type, args)
        full_img_list.extend(img_list)

    if args.save_images:
        with open(out_json_path, "w+") as json_file:
            json.dump(full_img_list, json_file, ensure_ascii=False, indent=4)
        create_tar(args.dataset_name, out_json_path, args.intermediate_dir, args.tar_dir, args.final_dir)


if __name__ == "__main__":
    main()
