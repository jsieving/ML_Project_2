'''
This contains the Impression and ColorImpression classes, which store layered conditional probabilities of a set of images. This enables images to be
generated which have similar conditional probabilities between pixel layers.

The results are somewhat "impressionistic" remixes of the input data, with good
locality and color matching but blurry, noisy details.

Author: Jane Sieving
'''

import numpy as np
import time
from random import randint
from datetime import datetime
import pickle
from pprint import PrettyPrinter
import PIL.Image as pim

p = PrettyPrinter()

class Impression:
    def __init__(self, size = 4, buckets = 8):
        self.img_count = 0
        self.size = size
        self.buckets = buckets
        self.depth = int(np.log2(size) + 1)
        self.levels = {}
        self.levels[0] = np.zeros((buckets))
        self.dirs = [(0,0), (0,1), (0,-1), (-1,0), (1,0)]
        for i in range(1, self.depth):
            self.levels[i] = {}

        for depth, level in self.levels.items():
            if depth == 0:
                continue
            for i in range(2**depth):
                for j in range(2**depth):
                    refs = {}
                    for dir in self.dirs:
                        row = i // 2 + dir[0]
                        col = j // 2 + dir[1]
                        if (0 <= row < 2**(depth-1)) and (0 <= col < 2**(depth-1)):
                            refs[dir] = np.zeros((buckets, buckets))
                    level[(i, j)] = refs

    def print_lvl(self, n):
        if n == -1:
            n = self.depth - 1
        if n < self.depth:
            if n > 0:
                level = self.levels[n]
                viz = np.zeros((2**n, 2**n))
                for coord, context in level.items():
                    for dir in context.keys():
                        viz[coord] += 1
                p.pprint(viz)
            else:
                print(self.levels[0])
        else:
            print("No such level:", n)

    def remember(self, img):
        if img.shape != (self.size, self.size):
            print("Image incorrect size:", img.shape)

        self.img_count += 1
        averages = [img]
        for d in range(1, self.depth):
            size = int(self.size/(2**d))
            average = np.zeros((size, size))
            prev = averages[d-1]
            for i in range(size):
                for j in range(size):
                    sum = prev[2*i, 2*j] + prev[2*i, 2*j+1] + prev[2*i+1, 2*j] + prev[2*i+1, 2*j+1]
                    average[i, j] = sum
            average //= 4
            averages.append(average)

        for d, level in self.levels.items():
            # d is the depth
            avg = averages[self.depth-d-1]
            if d == 0:
                level[int(avg[0,0])] += 1
            else:
                prev = averages[self.depth-d]
                for coord, context in level.items():
                    for dir, L_dists in context.items():
                        row = coord[0] // 2 + dir[0]
                        col = coord[1] // 2 + dir[1]
                        data = int(prev[row, col])
                        truth = int(avg[coord])
                        L_dists[data, truth] += 1

    def resize(self, arr, size=None):
        if size is None:
            size = self.size
        img = pim.fromarray(arr)
        new = pim.Image.resize(img, (size, size))
        return np.asarray(new)

    def halfsample(self, arr):
        size = int(arr.shape[0]/2)
        result = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                sum = arr[2*i, 2*j] + arr[2*i, 2*j+1] + arr[2*i+1, 2*j] + arr[2*i+1, 2*j+1]
                result[i, j] = sum
        result //= 4
        return result

    def imagine(self, prev = None):
        if prev is not None:
            depth = int(np.log2(prev.shape[0]) + 1)
        else:
            if self.levels[0].sum() == 0:
                color = np.random.choice(np.arange(0, self.buckets))
            else:
                probs = self.levels[0] / self.img_count
                color = np.random.choice(np.arange(0, self.buckets), p=probs)
            prev = np.asarray([[color]])
            depth = 1
        image = prev # catches when depth is already >= self.depth

        while depth < self.depth:
            image = np.zeros((2**depth, 2**depth))
            level = self.levels[depth]
            for coord, context in level.items():
                probability_dist = np.full((self.buckets), 1/self.buckets)

                for dir, L_dists in context.items():
                    row = coord[0] // 2 + dir[0]
                    col = coord[1] // 2 + dir[1]
                    data = int(prev[row, col]) # color of known data
                    likely_dist = L_dists[data] # likelihood of colors given this data
                    total_L = likely_dist.sum()
                    if total_L > 0:
                        probability_dist *= likely_dist / total_L

                total_P = probability_dist.sum()
                if total_P > 0: # ensure likelihoods have not multiplied this to 0
                    probability_dist /= total_P
                    color = np.random.choice(np.arange(0, self.buckets), p=probability_dist)
                else: # if this is an unseen set of data, choose randomly
                    color = np.random.choice(np.arange(0, self.buckets))
                image[coord] = color
            prev = image
            depth += 1
        return image

    def save_image(self, arr, name = None, folder = "output/", ext = ".png"):
        arr *= 256/self.buckets
        data = arr.astype('uint8')
        im = pim.fromarray(data)

        if name is None:
            time = datetime.now() # Get the current time
            # make a name by formatting the current time with the extension
            name = time.strftime("%Y-%m-%d_%H-%M-%S")

        try:
            im.save(folder + name + ext) # Save the image to the given folder
            print("File saved as %s in %s" % (name+ext, folder))
        except:
            print("Could not save image:", folder + name + ext)

    def save_impression(self, name = None, folder = "learning/"):
        ext = ".pkl"
        if name is None:
            time = datetime.now()
            timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
            info = "BW_%is_%ib_%iim__" % (self.size, self.buckets, self.img_count)
            name = info + timestr

        try:
            file = open(folder+name+ext, 'wb')
            pickle.dump(self, file)
            file.close()
            print("File saved as %s in %s" % (name+ext, folder))
        except:
            print("Could not save object:", folder + name+ext)


class ColorImpression:
    def __init__(self, size = 4, buckets = 8):
        self.img_count = 0
        self.size = size
        self.buckets = buckets
        self.depth = int(np.log2(size) + 1)
        self.levels = {}
        self.levels[0] = np.zeros((3, buckets))
        self.dirs = [(0,0,0), (0,1,0), (0,-1,0), (-1,0,0), (1,0,0), (0,0,-1), (0,0,1)]
        for i in range(1, self.depth):
            self.levels[i] = {}

        for depth, level in self.levels.items():
            if depth == 0:
                continue
            for i in range(2**depth):
                for j in range(2**depth):
                    for k in range(3):
                        refs = {}
                        for dir in self.dirs:
                            row = i // 2 + dir[0]
                            col = j // 2 + dir[1]
                            chan = k + dir[2]
                            if (0 <= row < 2**(depth-1)) and (0 <= col < 2**(depth-1)):
                                refs[dir] = np.zeros((buckets, buckets))
                        level[(i, j, k)] = refs

    def print_lvl(self, n):
        if n == -1:
            n = self.depth - 1
        if n < self.depth:
            if n > 0:
                level = self.levels[n]
                viz = np.zeros((2**n, 2**n))
                for coord, context in level.items():
                    for dir in context.keys():
                        viz[coord] += 1
                p.pprint(viz)
            else:
                print(self.levels[0])
        else:
            print("No such level:", n)

    def remember(self, img):
        if img.shape != (self.size, self.size, 3):
            print("Image incorrect size:", img.shape)

        self.img_count += 1
        averages = [img]
        for d in range(1, self.depth):
            size = int(self.size/(2**d))
            average = np.zeros((size, size, 3))
            prev = averages[d-1]
            for i in range(size):
                for j in range(size):
                    # sure hopes this works in 3d
                    sum = prev[2*i, 2*j] + prev[2*i, 2*j+1] + prev[2*i+1, 2*j] + prev[2*i+1, 2*j+1]
                    average[i, j] = sum
            average //= 4
            averages.append(average)

        for d, level in self.levels.items():
            # d is the depth
            avg = averages[self.depth-d-1]
            if d == 0: # If at lowest-res level
                rgb = avg[0,0] # The RGB value of that one pixel
                for i in range(3): # for each channel
                    # increase the count of that color in that channel by 1
                    level[i][int(rgb[i])] += 1
            else:
                prev = averages[self.depth-d]
                for coord, context in level.items():
                    for dir, L_dists in context.items():
                        row = coord[0] // 2 + dir[0] # get adjacent pixels, at lower resolution layer
                        col = coord[1] // 2 + dir[1]
                        chan = (coord[2] + dir[2]) % 3 # get adjacent channels
                        data = int(prev[row, col, chan])
                        truth = int(avg[coord])
                        L_dists[data, truth] += 1

    def resize(self, arr, size=None):
        if size is None:
            size = self.size
        img = pim.fromarray(arr)
        new = pim.Image.resize(img, (size, size))
        return np.asarray(new)

    def halfsample(self, arr):
        size = int(arr.shape[0]/2)
        result = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                sum = arr[2*i, 2*j] + arr[2*i, 2*j+1] + arr[2*i+1, 2*j] + arr[2*i+1, 2*j+1]
                result[i, j] = sum
        result //= 4
        return result

    def imagine(self, prev = None):
        if prev is not None: # If there's a low-resolution starting point
            depth = int(np.log2(prev.shape[0]) + 1)
        else: # starting from scratch
            if self.levels[0].sum() == 0: # If there's no past data for the first level, choose a random color
                color = np.random.choice(np.arange(0, self.buckets), size=3)
            else:
                probs = self.levels[0] / self.img_count ################################################## you got this far
                color = np.zeros(3)
                for i in range(3): # pick each color channel
                    color[i] = np.random.choice(np.arange(0, self.buckets), p=probs[i])
            prev = np.asarray([[color]])
            depth = 1
        image = prev # catches when depth is already >= self.depth

        while depth < self.depth:
            image = np.zeros((2**depth, 2**depth, 3))
            level = self.levels[depth]
            for coord, context in level.items():
                probability_dist = np.full((self.buckets), 1/self.buckets)

                for dir, L_dists in context.items():
                    row = coord[0] // 2 + dir[0]
                    col = coord[1] // 2 + dir[1]
                    chan = (coord[2] + dir[2]) % 3
                    data = int(prev[row, col, chan]) # color of known data
                    likely_dist = L_dists[data] # likelihood of colors given this data
                    total_L = likely_dist.sum()
                    if total_L > 0:
                        probability_dist *= likely_dist / total_L

                total_P = probability_dist.sum()
                if total_P > 0: # ensure likelihoods have not multiplied this to 0
                    probability_dist /= total_P
                    color = np.random.choice(np.arange(0, self.buckets), p=probability_dist)
                else: # if this is an unseen set of data, choose randomly
                    color = np.random.choice(np.arange(0, self.buckets))
                image[coord] = color
            prev = image
            depth += 1
        return image

    def save_image(self, arr, name = None, folder = "output/", ext = ".png"):
        arr *= 256/self.buckets
        data = arr.astype('uint8')
        im = pim.fromarray(data)

        if name is None:
            time = datetime.now() # Get the current time
            # make a name by formatting the current time with the extension
            name = time.strftime("%Y-%m-%d_%H-%M-%S")

        try:
            im.save(folder + name + ext) # Save the image to the given folder
            print("File saved as %s in %s" % (name+ext, folder))
        except:
            print("Could not save image:", folder + name + ext)

    def save_impression(self, name = None, folder = "learning/"):
        ext = ".pkl"
        if name is None:
            time = datetime.now()
            timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
            info = "C_%is_%ib_%iim__" % (self.size, self.buckets, self.img_count)
            name = info + timestr

        try:
            file = open(folder+name+ext, 'wb')
            pickle.dump(self, file)
            file.close()
            print("File saved as %s in %s" % (name+ext, folder))
        except:
            print("Could not save object:", folder + name+ext)
#------------------------------------------------------------------------------#

def gen_image(size = 4, buckets = 8, color = False):
    if color:
        img = np.ndarray((size, size, 3))
    else:
        img = np.ndarray((size, size))
    for i in range(size):
        for j in range(size):
            if color:
                for k in range(3):
                    img[i, j, k] = randint(0, buckets-1)
            else:
                img[i, j] = randint(0, buckets-1)
    return img

def load_impression(filename):
    f = open(filename, 'rb')
    impression = pickle.load(f)
    print("Loaded", filename)
    return impression

#------------------------------------------------------------------------------#

if __name__ == "__main__":
    n = 100
    b = 16
    s = 64

    # Timing test
    imp = ColorImpression(size=s, buckets=b)
    imgs = []
    for i in range(n):
        imgs.append(gen_image(size=s, buckets=b, color=True))

    start_time = time.clock()
    for img in imgs:
        imp.remember(img)
    total_time = time.clock() - start_time

    print("\t%i buckets, size %i:\t%.4f total\t%.8f per item" % (b, s, total_time, total_time/n))
    imp.save_impression()

#------------------------------------------------------------------------------#

    # Sanity test - 1 remembered image should be able to be reproduced
    imp2 = ColorImpression(size=s, buckets=b)
    img1 = gen_image(size=s, buckets=b, color=True)
    imp2.remember(img1)
    imp2.save_image(img1)
    input()
    img2 = imp2.imagine()
    imp2.save_image(img2)
