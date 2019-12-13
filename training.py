'''
Creates an "impression" of a set of images, optionally separated by class, in
order to generate new images which are probabilistically derived from the
original data. It saves the probability data to a file in the `learning` folder.
After this is run, `generate.py` must be run to actually generate images.

Dataset can be provided as one of the names 'flowers', 'faces', and 'rps'.
The other command line arguments (which are optional but position-specific) are
resolution (must be a power of 2), number of buckets to divide color into, and
whether or not to use color. So, `python3 training.py flowers 64 16 c` produces
a 64x64 color impression of flowers, with RGB values each going from 0-15.

For large datasets, if you want to get some results quickly you can pass a final value as an argument that allows you to only process 1 out of N images. So,
`python3 training.py faces 64 16 x 5` will process every fifth image, generating
a 64x64 grayscale impression of the images. 

Author: Jane Sieving

---
Sources:

Loading data: https://www.tensorflow.org/tutorials/load_data/images

Flowers dataset: https://www.tensorflow.org/datasets/catalog/tf_flowers

RPS dataset: http://www.laurencemoroney.com/rock-paper-scissors-dataset/

Faces data (Labeled Faces in the Wild) references:

[1] Lior Wolf, Tal Hassner, and Yaniv Taigman, Effective Face Recognition by
Combining Multiple Descriptors and Learned Background Statistics, IEEE Trans. on
Pattern Analysis and Machine Intelligence (TPAMI), 33(10), Oct. 2011

[2] Lior Wolf, Tal Hassner and Yaniv Taigman, Similarity Scores based on
Background Samples, Asian Conference on Computer Vision (ACCV), Xi' an, Sept 2009

[3] Yaniv Taigman, Lior Wolf and Tal Hassner, Multiple One-Shots for Utilizing
Class Label Information, The British Machine Vision Conference (BMVC), London,
Sept 2009
'''

import numpy as np
import PIL.Image as pim
import tensorflow as tf
from glob import glob
import sys
from os import path
from impressions import *

if len(sys.argv) < 2: # make sure dataset is specified
    print("Please provide 'flowers', 'rps' or 'faces' as an argument to this program to select a dataset.")
    exit()

# Select data
if sys.argv[1] == "flowers":
    data_link = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
    data_name = 'flower_photos'
    separate_classes = True
    images_colored = True
elif sys.argv[1] == "rps":
    data_link = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip'
    data_name = 'rps'
    separate_classes = True
    images_colored = True
elif sys.argv[1] == "faces":
    data_link = 'https://drive.google.com/file/d/1p1wjaqpTh_5RHfJu4vUh8JJCdKwYMHCp/view?usp=sharing'
    data_name = 'lfw2'
    separate_classes = False
    images_colored = False
else:
    print("Please provide 'flowers', 'rps' or 'faces' as an argument to this program to select a dataset.")
    exit()

# Set other program options
if len(sys.argv) > 2:
    SIZE = int(sys.argv[2])
else:
    SIZE = 32
    print("Image size:", SIZE)

if len(sys.argv) > 3:
    BUCKETS = int(sys.argv[3])
else:
    BUCKETS = 16
    print("Color buckets:", BUCKETS)

if len(sys.argv) > 4:
    if str(sys.argv[4])[0].lower() == 'c':
        COLOR = True
else:
    COLOR = False
    print("Color images:", COLOR)

if len(sys.argv) > 5:
    skip_count = int(sys.argv[5])
else:
    skip_count = 0
    print("Skip over:", skip_count - 1)

# Download data if not already downloaded
data_dir = tf.keras.utils.get_file(origin=data_link, fname=data_name, untar=True)

images = glob(data_dir + "/*/*.jpg")

img_count = len(images)
img_progress = 0
print(img_count, "images loaded.")

class_dict = {}
if separate_classes:
    # create list of classes
    class_dirs = glob(data_dir + "/*")
    class_names = []
    for c in class_dirs:
        if path.isdir(c):
            class_names.append(path.relpath(c, data_dir))
    print("Classes found:", class_names)
    for name in class_names:
        class_dict[name] = glob(data_dir + "/%s/*.jpg" % name)
else: # put all contents together
    print("Combining all subdirectories/classes.")
    class_dict[data_name] = glob(data_dir + "/*/*.jpg")

for class_name, image_files in class_dict.items():
    if COLOR and images_colored:
        imp = ColorImpression(size=SIZE, buckets=BUCKETS)
        mode = "C"
    else:
        imp = Impression(size=SIZE, buckets=BUCKETS)
        mode = "BW"

    start_time = time.clock()
    for filename in image_files:
        # only process every `skip_count` images
        if img_progress % skip_count != 0:
            continue
        orig = pim.open(filename)
        sized = pim.Image.resize(orig, (SIZE, SIZE))
        arr = np.asarray(sized)
        # single out only grey channel if appropriate
        if not COLOR and images_colored:
            arr = arr[:, :, 0]
        arr = arr * float(BUCKETS) / 256. # squash color spectra into buckets
        imp.remember(arr)
        img_progress += 1
        if img_progress % 100 == 0:
            print("%.1f percent -\t%i/%i images remembered" % (img_progress/img_count, img_progress, img_count))
    total_time = time.clock() - start_time

    print("%i images in %.2f seconds" % (imp.img_count, total_time))

    imp.save_impression("%s_%is_%ib__%s" % (mode, imp.size, imp.buckets, class_name))
