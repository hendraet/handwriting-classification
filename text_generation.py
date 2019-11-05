import os
import string

from PIL import Image, ImageDraw, ImageFont
import random


def draw_string(txt):
    fontsize = 1
    # font = 'Pillow/Tests/fonts/DejaVuSans.ttf'
    font_dir = "google-fonts"
    font = font_dir + "/" + random.choice(os.listdir(font_dir))
    img_fraction = 0.90

    img = Image.new("RGB", (100, 42), (255, 255, 255))
    d = ImageDraw.Draw(img)

    img_font = ImageFont.truetype(font, fontsize)
    while img_font.getsize(txt)[0] < img_fraction * img.size[0] and img_font.getsize(txt)[1] < img_fraction * img.size[1]:
        fontsize += 1
        img_font = ImageFont.truetype(font, fontsize)

    fontsize -= 1
    img_font = ImageFont.truetype(font, fontsize)

    d.text((5, 5), txt, fill=(0, 0, 0), font=img_font)

    img.show()


def generate_date():
    day = str(random.randint(1, 31)).zfill(2)
    month = str(random.randint(1, 12)).zfill(2)
    year = str(random.randint(0, 99)).zfill(2) if random.randint(0, 1) is 0 else str(random.randint(1000, 2020))
    return ".".join([day, month, year])


def generate_text():
    num_chars = random.randint(4, 10)
    return "".join(random.choices(string.ascii_lowercase, k=num_chars))


if __name__ == "__main__":
    for i in range(1, 10):
        generated_string = generate_text()
        # generated_string = generate_date()
        draw_string(generated_string)
