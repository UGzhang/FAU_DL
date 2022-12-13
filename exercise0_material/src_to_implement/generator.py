import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

import random


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.

'''
The design thinking behind this class:
    1. read all file name in a list 
    2. read json file
    3. prepare array
        image -> [batchSize, imageSize, imageSize, channel]
        label -> [batchSize]
    5. batch size for loop : read image and label
'''
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.__index = 0  # record the index of current image
        self.__epoch = 0  # record the epoch

        self.__data_init()

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method

        # resume the index
        if self.__index >= len(self.image_name_list):
            self.__index = 0
            self.__epoch += 1

        # initial the shuffle operator
        if self.__index == 0 and self.shuffle:
            np.random.shuffle(self.image_name_list)

        images = np.empty([self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]])
        labels = np.empty(self.batch_size, dtype=np.integer)

        for i in range(self.batch_size):

            # get full path
            image_path = os.path.join(self.file_path, self.image_name_list[self.__index])

            # example: 0.npy -> 0
            # get the value of key 0 in dict
            label = self.labels_dict.get(self.image_name_list[self.__index].split('.')[0])

            # load the augment
            image = self.augment(np.load(image_path))

            labels[i] = label
            images[i] = image
            self.__index += 1

        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function

        if img.shape != self.image_size:
            img = np.resize(img, self.image_size)

        if self.mirroring:
            if random.getrandbits(1): # random 0 1
                img = np.fliplr(img)

        if self.rotation:
            img = np.rot90(img, random.getrandbits(2))  # random 0 1 2 3

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.__epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict.get(x)

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        cols = 3
        row = self.batch_size // cols + (1 if self.batch_size % cols else 0)
        imgs, labs = self.next()
        fig = plt.figure()
        for i in range(1, self.batch_size+1):
            lab = self.class_dict[labs[i-1]]
            fig.add_subplot(row, cols, i)
            plt.imshow(imgs[i-1].astype('uint8'))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.title(lab)
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        return 0

    def __data_init(self):
        # import all image file name
        # remove the  hidden file in MACOS
        self.image_name_list = [i for i in os.listdir(self.file_path) if i[0] != "."]

        # sort 0.npy 1.npy ...
        self.image_name_list.sort(key=lambda x: int(x.split('.')[0]))

        with open(self.label_path, 'r') as f:
            self.labels_dict = json.load(f)

        # computer number of image of last batch
        redundancy = len(self.labels_dict) % self.batch_size
        if redundancy != 0:
            # reuse the beginning of data set
            self.image_name_list.extend(self.image_name_list[:self.batch_size-redundancy])

