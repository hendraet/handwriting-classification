import json
import shutil

import os
import tarfile


def create_tar(new_dataset_name, new_dataset_description, img_dir, tar_dir, move_images_instead_of_copy=True,
               final_dir=None):
    tar_filename = os.path.join(tar_dir, new_dataset_name + ".tar.bz2")
    with open(new_dataset_description) as dd_f:
        samples = json.load(dd_f)
    image_files = [os.path.join(img_dir, sample["path"]) for sample in samples]

    num_image_files = len(image_files)
    with tarfile.open(tar_filename, "w:bz2") as tar:
        for i, filename in enumerate(image_files):
            if ((i + 1) % 1000) == 0:
                print(f"Adding file {i +1}/{num_image_files} to tar file")
            tar.add(filename, arcname=os.path.basename(filename))
        tar.add(new_dataset_description, arcname=os.path.basename(new_dataset_description))

    if final_dir is not None:
        for filename in image_files:
            if move_images_instead_of_copy:
                shutil.move(filename, final_dir)
            else:
                shutil.copy(filename, final_dir)
        shutil.move(new_dataset_description, final_dir)
