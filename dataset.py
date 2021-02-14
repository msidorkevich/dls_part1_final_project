import logging
import os
from os import listdir
from os.path import join

import numpy as np
import skimage.io


def load_images():
    logging.info("Extracting dataset")
    os.system("unzip -o dataset.zip")

    images = []
    for filename in listdir("./dataset"):
        image = skimage.io.imread(join("./dataset", filename))
        images.append(image)

    return np.stack(images)