'''
Generates 5 images from each class of the specified dataset.

Requires that `training.py` be run beforehand to generate the "impression"
needed to generate images. It should be run with the same command line arguments
since it needs to use a saved Impression file with compatible dimensions.

Dataset can be provided as the folder name of a dataset in `~/.keras/datasets`.
The other command line arguments (which are optional but position-specific) are
resolution (must be a power of 2), number of buckets to divide color into, and
whether or not to use color. So, `python3 generate.py flowers 64 16 c` produces
64x64 color representations of flowers with RGB values each going from 0-15.

Author: Jane Sieving
'''

from glob import glob
import sys
from os import path
from impressions import *

dataset_base_path = path.expanduser("~/.keras/datasets/")

print(len(sys.argv))
print(dataset_base_path)

# Select data
if len(sys.argv) > 1:
    folder_name = str(sys.argv[1])
    data_full_path = path.join(dataset_base_path, folder_name)
    if folder_name.lower() == "lfw2": # Oof hard coding oh well
        separated_classes = False
    else:
        separated_classes = True
else:
    print("Please provide the folder name for a dataset. Current folders in `~/.keras/datasets`:")
    dirs = glob(dataset_base_path + '*')
    for dir in dirs:
        print(path.relpath(dir, dataset_base_path))
    exit()

if not path.isdir(data_full_path):
    print("Please provide a valid folder name for a dataset. Current folders in `~/.keras/datasets`:")
    dirs = glob(dataset_base_path + '*')
    for dir in dirs:
        print(path.relpath(dir, dataset_base_path))
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

# create list of classes
if separated_classes:
    class_dirs = glob(data_full_path + "/*")
    class_names = []
    for c in class_dirs:
        if path.isdir(c):
            class_names.append(path.relpath(c, data_full_path))
else:
    class_names = [folder_name]

if COLOR:
    mode = "C"
else:
    mode = "BW"

for name in class_names:
    imp = load_impression("learning/%s_%is_%ib__%s.pkl" % (mode, SIZE, BUCKETS, name))

    for i in range(5): # creates 5 images in each class
        img = imp.imagine()
        imp.save_image(img, name = "%s_%s_%i_%i_%i" % (mode, name, SIZE, BUCKETS, i))
