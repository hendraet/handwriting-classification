import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')


def remove_black_rect(img):
    tmp = img[:, :, 0]  # only ok if image really grayscale
    summed_columns = np.sum(tmp, axis=0)
    first_col = np.searchsorted(summed_columns, 0, side='right')
    last_col = np.searchsorted(summed_columns[::-1], 0, side='right')

    if (first_col + last_col) < len(summed_columns):
        return img[:, first_col:-(last_col + 1), :]
    else:
        return img


if __name__ == '__main__':
    image_path = None
    assert(image_path is not None)
    gs_img = np.array(Image.open(image_path))
    # Image.fromarray(gs_img, 'L').show()
    stacked_img = np.stack((gs_img,) * 3, axis=-1)

    cropped_array = remove_black_rect(stacked_img)

    cropped_img = Image.fromarray(cropped_array, 'RGB')
    cropped_img.show()
