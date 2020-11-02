import os

import cv2

out_dir = "../../orig_datasets/WPI/binarised_dataset"
in_dir = "../../orig_datasets/WPI/labeled_dataset/"

for img_path in os.listdir(in_dir):
    full_img_path = os.path.join(in_dir, img_path)
    if os.path.splitext(img_path)[1] != ".png":
        continue

    # img_path = "010000006765822_007_cette.png"

    img = cv2.imread(full_img_path, 0)
    # gauss_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imwrite(os.path.join(out_dir, f"{img_name}_gauss.png"), gauss_th)

    # _, otsus_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite(os.path.join(out_dir, f"{img_name}_otsus.png"), otsus_th)

    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    _, otsus_th_blur = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    # cv2.imwrite(os.path.join(out_dir, f"{img_name}_otsus_blur.png"), otsus_th_blur)
    cv2.imwrite(os.path.join(out_dir, img_path), otsus_th_blur)
