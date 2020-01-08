import argparse
from datetime import datetime
import json
import time

import nltk
import os
import random

from PIL import Image, ImageDraw, ImageFont
from nltk.corpus import words


def generate_number():
    if bool(random.getrandbits(1)):
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
    date_formats = ['%d.%m.%y', '%d.%m.%Y', '%m/%d/%y', '%m/%d/%Y', '%d. %B %Y', '%d. %b %Y', '%B %y', '%d. %B',
                    '%d. %b']

    start = datetime.strptime('01.01.1000', '%d.%m.%Y')
    end = datetime.strptime('01.01.2020', '%d.%m.%Y')
    delta = end - start

    rand_date = start + delta * random.random()

    return rand_date.strftime(random.choice(date_formats))


def generate_text(word_list):
    return random.choice(word_list)


def generate_image(txt):
    fontsize = 30
    font_dir = 'google-fonts'
    font_name = random.choice(os.listdir(font_dir))
    font = font_dir + '/' + font_name
    target_dimensions = (350, 60)

    img_font = ImageFont.truetype(font, fontsize)
    text_dimensions = img_font.getsize(txt)

    img = Image.new('RGB', text_dimensions, (255, 255, 255))  # TODO: save as grayscale
    d = ImageDraw.Draw(img)

    d.text((0, 0), txt, fill=(0, 0, 0), font=img_font)

    ratio = target_dimensions[1] / text_dimensions[1]
    new_dimensions = tuple([int(dim * ratio) for dim in text_dimensions])
    resized_img = img.resize(new_dimensions)
    padded_img = Image.new('RGB', target_dimensions, (0, 0, 0))
    padded_img.paste(resized_img, ((target_dimensions[0] - new_dimensions[0]) // 2,
                                   (target_dimensions[1] - new_dimensions[1]) // 2))

    return padded_img, font_name, text_dimensions


def main():
    assert False, "Removed TODO so that generated images are only grayscale?"
    parser = argparse.ArgumentParser(description='Generating images with random text')
    parser.add_argument('-d', '--dir', type=str, default='prep/datasets/',
                        help='path to directory where images should be stored')
    parser.add_argument('-n', '--num', type=int, default=1, help='num of images to be generated')
    parser.add_argument('-t', '--type', type=str, choices=['date', 'text', 'num'], default='date',
                        help='type of text to be generated')
    parser.add_argument('-s', '--show', action='store_true', help='show image(s) after generation')
    parser.add_argument('-p', '--json_img_path', type=str, help='dir path that should be written to json')
    parser.add_argument('--save', action='store_true', help='save image(s) after generation')
    args = parser.parse_args()

    string_type = args.type
    show_image = args.show
    save_image = args.save
    img_dir = args.dir
    json_img_dir = args.json_img_path
    json_path = img_dir + 'data' + '.json'
    img_list = []

    if string_type == 'text':
        # installation of word list is needed on first installation of nltk.
        # nltk.download('words')
        max_length = 9
        word_list = words.words()
        word_list = [word for word in word_list if len(word) < max_length]

    for i in range(0, args.num):
        if string_type == 'date':
            generated_string = generate_date()
        elif string_type == 'text':
            generated_string = generate_text(word_list)
        elif string_type == 'num':
            generated_string = generate_number()
        else:
            print('Unexpected type')
            exit(1)

        img, font_name, text_dimensions = generate_image(generated_string)
        cleansed_string = generated_string.replace("/", "_").replace(" ", "_")
        img_name = cleansed_string + '-' + os.path.splitext(font_name)[0]
        img_path = img_dir + img_name + '.png'
        json_img_path = json_img_dir + img_name + '.png'

        if show_image:
            img.show()

        if save_image:
            with open(img_path, 'wb+') as img_file:
                img.save(img_file, format='PNG')

        info = {
            'string': generated_string,
            'type': string_type,
            'path': json_img_path,
            'font': font_name
        }
        img_list.append(info)

    print(img_list)
    if save_image:
        with open(json_path, 'w+') as json_file:
            json.dump(img_list, json_file, indent=4)


if __name__ == '__main__':
    main()
