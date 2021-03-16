import os

import cv2
import numpy
from PIL import Image


def binarise_opencv_image(image: numpy.ndarray) -> numpy.ndarray:
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    _, binarised_image = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarised_image


def pil_image_to_opencv(pil_image: Image.Image) -> numpy.ndarray:
    return cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)


def opencv_image_to_pil(opencv_image: numpy.ndarray) -> Image.Image:
    return Image.fromarray(opencv_image)


def binarise_pil_image(image: Image.Image) -> Image.Image:
    binarised_image = binarise_opencv_image(pil_image_to_opencv(image))
    return opencv_image_to_pil(binarised_image)


def debug():
    in_dir = "../../../web/static/examples/classification/wpi"
    out_dir = "../../../web/static/examples/tmp"

    for img_path in os.listdir(in_dir)[:10]:
        full_img_path = os.path.join(in_dir, img_path)
        if os.path.splitext(img_path)[1] != ".png":
            continue

        # img_path = "010000006765822_007_cette.png"

        image = cv2.imread(full_img_path, 0)
        otsus_th_blur = binarise_opencv_image(image)

        combined_image = numpy.concatenate((image, otsus_th_blur))
        # cv2.imwrite(os.path.join(out_dir, img_path), otsus_th_blur)
        opencv_image_to_pil(combined_image).show()


if __name__ == '__main__':
    debug()
