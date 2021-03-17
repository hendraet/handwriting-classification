import json
import os
import random

from PIL import Image

from handwriting_embedding.predict import HandwritingClassifier


def debug():
    # ------------ 5CHPT ----------------
    # --- Taking random samples from json ---
    # num_samples = 100
    # image_dir = "/datasets/train"
    # with open(os.path.join(image_dir, "5CHPT_plus_unlabelled_train.json")) as json_f:
    #     file_list = [f["path"] for f in json.load(json_f) if "font" in f and not (f["type"] == "alpha_num" or
    #                                                                           f["type"] == "plz")]
    # images = random.sample(file_list, num_samples)

    # --- Take all samples from a dir ---
    # image_dir = "/datasets/5CHPT_plus_unlabelled/tmp"
    # image_dir = "/datasets/wpi_demo/5CHPT"
    # images = os.listdir(image_dir)
    #
    # for filename in images:
    #     image_path = os.path.join(image_dir, filename)
    #     image = Image.open(image_path)
    #
    #     predictor = HandwritingClassifier()
    #     prediction_result = predictor.predict_image(image)
    #     print(f"{filename}: {prediction_result}")

    # --------- Select WPI samples --------------------
    image_dir = "/datasets/wpi_orig"
    with open(os.path.join(image_dir, "wpi_words_dates_nums_alphanum.json")) as json_f:
        file_list = [(f["path"], f["type"]) for f in json.load(json_f)]

    long_class_label_dict = {
        "alpha_num": "Alphanumeric",
        "alphanum": "Alphanumeric",
        "date": "Date",
        "num": "Number",
        "plz": "Zip Code",
        "text": "Word"
    }

    for filename, label in file_list:
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)

        predictor = HandwritingClassifier()
        prediction_result = predictor.predict_image(image)

        if long_class_label_dict[label] == prediction_result["predicted_class"]:
            print(f"{filename}: {prediction_result}")


if __name__ == '__main__':
    debug()
