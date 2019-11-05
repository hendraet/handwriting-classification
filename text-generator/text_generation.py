import argparse
import os
import string

from PIL import Image, ImageDraw, ImageFont
import random


def draw_string(txt, directory, show_image):
    fontsize = 1
    font_dir = 'google-fonts'
    font_name = random.choice(os.listdir(font_dir))
    font = font_dir + '/' + font_name
    img_fraction = 0.90
    img_name = txt + '-' + os.path.splitext(font_name)[0]

    img = Image.new('RGB', (100, 42), (255, 255, 255))
    d = ImageDraw.Draw(img)

    img_font = ImageFont.truetype(font, fontsize)
    while img_font.getsize(txt)[0] < img_fraction * img.size[0] and img_font.getsize(txt)[1] < img_fraction * img.size[1]:
        fontsize += 1
        img_font = ImageFont.truetype(font, fontsize)

    fontsize -= 1
    img_font = ImageFont.truetype(font, fontsize)

    d.text((5, 5), txt, fill=(0, 0, 0), font=img_font)

    if show_image:
        img.show()

    img.save(directory + img_name + '.png', format='PNG')


def generate_date():
    day = str(random.randint(1, 31)).zfill(2)
    month = str(random.randint(1, 12)).zfill(2)
    year = str(random.randint(0, 99)).zfill(2) if random.randint(0, 1) is 0 else str(random.randint(1000, 2020))
    return '.'.join([day, month, year])


def generate_text():
    num_chars = random.randint(4, 10)
    return ''.join(random.choices(string.ascii_lowercase, k=num_chars))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating images with random text')
    parser.add_argument('-d', '--dir', type=str, default='text-generator/generated-images/',
                        help='path to directory where images should be stored')
    parser.add_argument('-n', '--num', type=int, default=1, help='num of images to be generated')
    parser.add_argument('-t', '--type', type=str, default='date', help='type of text to be generated')
    parser.add_argument('-s', '--show', action='store_true', help='show image after generation')
    args = parser.parse_args()

    for i in range(0, args.num):
        if args.type is 'date':
            generated_string = generate_date()
        else:
            generated_string = generate_text()

        draw_string(generated_string, args.dir, args.show)
