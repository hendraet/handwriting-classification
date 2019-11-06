import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont
from nltk.corpus import words


def generate_date():
    day = str(random.randint(1, 31)).zfill(2)
    month = str(random.randint(1, 12)).zfill(2)
    year = str(random.randint(0, 99)).zfill(2) if random.randint(0, 2) is 0 else str(random.randint(1000, 2020))
    return '.'.join([day, month, year])


def generate_text():
    # installation of word list is needed on first installation of nltk. Just execute nltk.download() and select the
    # "words" corpus
    word_list = words.words()
    return random.choice(word_list)


def generate_image(txt):
    fontsize = 1
    font_dir = 'google-fonts'
    font_name = random.choice(os.listdir(font_dir))
    font = font_dir + '/' + font_name
    img_fraction = 0.90

    img = Image.new('RGB', (100, 42), (255, 255, 255))
    d = ImageDraw.Draw(img)

    img_font = ImageFont.truetype(font, fontsize)
    while img_font.getsize(txt)[0] < img_fraction * img.size[0] and \
            img_font.getsize(txt)[1] < img_fraction * img.size[1]:
        fontsize += 1
        img_font = ImageFont.truetype(font, fontsize)

    fontsize -= 1
    img_font = ImageFont.truetype(font, fontsize)

    d.text((5, 5), txt, fill=(0, 0, 0), font=img_font)

    return img, font_name


def main():
    parser = argparse.ArgumentParser(description='Generating images with random text')
    parser.add_argument('-d', '--dir', type=str, default='text-generator/generated-images/',
                        help='path to directory where images should be stored')
    parser.add_argument('-n', '--num', type=int, default=1, help='num of images to be generated')
    parser.add_argument('-t', '--type', type=str, choices=['date', 'text'], default='date', help='type of text to be '
                                                                                                 'generated')
    parser.add_argument('-s', '--show', action='store_true', help='show image(s) after generation')
    parser.add_argument('--save', action='store_true', help='save image(s) after generation')
    args = parser.parse_args()

    string_type = args.type
    show_image = args.show
    save_image = args.save
    img_dir = args.dir
    json_path = img_dir + 'data' + '.json'
    img_list = []

    for i in range(0, args.num):
        if string_type == 'date':
            generated_string = generate_date()
        elif string_type == 'text':
            generated_string = generate_text()
        else:
            print('Unexpected type')
            exit(1)

        img, font_name = generate_image(generated_string)
        img_name = generated_string + '-' + os.path.splitext(font_name)[0]
        img_path = img_dir + img_name + '.png'

        if show_image:
            img.show()

        if save_image:
            img.save(img_path, format='PNG')

        info = {
            'string': generated_string,
            'type': string_type,
            'path': img_path,
            'font': font_name
        }
        img_list.append(info)

    print(img_list)
    if save_image:
        with open(json_path, 'w+') as json_file:
            json.dump(img_list, json_file, indent=4)


if __name__ == '__main__':
    main()
