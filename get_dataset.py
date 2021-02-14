import logging
import os

import numpy as np
import skimage.io
from skimage.transform import resize


def fetch_dataset(images_name="lfw-deepfunneled",
                  dx=80, dy=80,
                  dimx=48, dimy=48):

    # download if not exists
    if not os.path.exists(images_name):
        logging.info("images not found, downloading...")
        os.system("wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz -O tmp.tgz")
        logging.info("extracting...")
        os.system("tar xvzf tmp.tgz")
        os.remove("tmp.tgz")
        logging.info("done extracting")
        assert os.path.exists(images_name)

    # read photos
    logging.info("preparing photo paths")
    photo_paths = []
    for dirpath, dirnames, filenames in os.walk(images_name):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath, fname)
                photo_paths.append(fpath)

    logging.info("reading photos")
    # image preprocessing
    images = []
    for photo_path in photo_paths:
        image = skimage.io.imread(photo_path)
        image = image[dy:-dy, dx:-dx]
        image = resize(image, [dimx, dimy])
        images.append(image)

    all_photos = np.stack(images)
    
    return all_photos
