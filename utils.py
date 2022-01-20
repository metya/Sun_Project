import copy
import os
import pickle
import time
import urllib
import warnings

import matplotlib.pyplot as plt
import numpy as np
# import pretrainedmodels

from astropy.io import fits
# from PIL import Image
from skimage.transform import rescale, resize
# from sklearn.metrics import f1_score
# from sklearn.utils import class_weight

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
import albumentations as alb

from torch.optim import lr_scheduler
from torchvision import datasets, models


class SunRegionDataset(data_utils.Dataset):
    def __init__(self, path_to_df_pkl, path_to_fits_folder, height, width,
                 only_first_class=False, transformations=None, logarithm=True, max=None):
        """
        Args:
            path_to_df_pkl (string): path or url to pkl file represents pandas dataframe with labels
            path_to_image_folder (string): path to folder with fits
            height (int): image height
            width (int): image width
            only_first_class (bool): create dataset with only one letter represents first layer of Mctosh classes
            transformation: pytorch transforms for transforms and tensor conversion
        """
        if path_to_df_pkl.startswith('http'):
            with urllib.request.urlopen(path_to_df_pkl) as pkl:
                self.sunspots = pickle.load(pkl)
        else:
            self.sunspots = pickle.load(path_to_df_pkl)
        self.classes = np.asarray(self.sunspots.iloc[:, 2].unique())
        self.height = height
        self.width = width
        self.folder_path, self.dirs, self.files = next(os.walk(path_to_fits_folder))
        self.len = len(self.files)
        self.ind = list(range(self.len))
        self.transformations = transformations
        self.alb_transorms = alb.Compose([
                                        alb.RandomRotate90(p=0.1),
                                        alb.Rotate(75, p=0.1),
                                        alb.Resize(224, 224, p=0.1),
                                        alb.RandomCrop(200, 200, p=0.1),
                                        alb.HorizontalFlip(),
                                        # alb.Transpose(),
                                        alb.VerticalFlip(),
                                        alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
                                        ], p=0.7)
        self.to_tensor = transforms.ToTensor()
        self.only_first_class = only_first_class
        self.height = height
        self.width = width
        self.logarithm = logarithm
        self.first_classes = set([class_[0] for class_ in self.sunspots['class'].unique()])
        self.second_classes = set([class_[1] for class_ in self.sunspots['class'].unique()])
        self.third_classes = set([class_[2] for class_ in self.sunspots['class'].unique()])
        if max == None:
            self.max = self.find_max_dataset()
        else:
            self.max = max

    def __getitem__(self, index):
        file_path = os.path.join(self.folder_path, self.files[index])
        with fits.open(file_path) as fits_file:
            data = fits_file[0].data

        if self.transformations is None:
            if self.logarithm:
                data = self.log_normalize(data)
            data = self.normalize_data(data)
            # data = resize(data, (self.height, self.width), anti_aliasing=True)
            data = self.aug()(image=data)['image']
            data = self.to_tensor(data).float()  # uncomment for float
            # data = data.repeat(3,1,1) # convert to 3 channels to use pretrein models
        else:
            data = self.transformations(data)

        mc_class = self.get_attr_region(self.files[index], self.sunspots, self.only_first_class)

        for ind, letter in enumerate(sorted(self.first_classes)):
            if letter == mc_class:
                num_class = ind

        return (data, num_class, mc_class)

    def __len__(self):
        return self.len

    def show_region(self, index):
        '''Plot region by index from dataset
        index: int, index of sample from dataset
        '''
        date, region = self.files[index].split('.')[1:3]
        file_path = os.path.join(self.folder_path, self.files[index])
        with fits.open(file_path) as fits_file:
            data = fits_file[0].data
        class_, size, location, number_ss = self.get_attr_region(self.files[index],
                                                                 self.sunspots,
                                                                 only_first_class=False,
                                                                 only_class=False)
        ax = plt.axes()
        ax.set_title(
            'Region {} on date {} with class {} on location {} with size {} and number_of_ss {}'
            .format(region, date, class_, location, size, number_ss))
        ax.imshow(data)
        # ax.annotate((24,12))

    def get_attr_region(self, filename, df, only_first_class=False, only_class=True):
        date, region = filename.split('.')[1:3]
        reg_attr = df.loc[date[:-7], int(region[2:])]
        if only_first_class:
            return reg_attr['class'][0]
        elif (not only_class) and (only_first_class):
            class_, \
                size, \
                location, \
                number_ss = reg_attr[['class', 'size', 'location', 'number_of_ss']]
            return class_[0], size, location, number_ss
        elif (not only_class) and (not only_first_class):
            return reg_attr[['class', 'size', 'location', 'number_of_ss']]
        else:
            return reg_attr['class']

    def log_normalize(self, data):
        return np.sign(data)*np.log1p(np.abs(data))

    def normalize_data(self, data):
        return data/self.max

    def find_max_dataset(self):
        m = []
        for file in self.files:
            with fits.open(self.folder_path + file) as ff:
                m.append(np.nanmax(np.abs(ff[0].data)))
        return np.max(m)
    
    def aug(self):
        return self.alb_transorms

    def split_dataset(self, val_size=None, test_size=None):
        '''Spliting dataset in optional test, train, val datasets
        test_size (optional): float from 0 to 1.
        val_size (optional): float from 0 to 1.
        shuffle (optional): bool, for shuffled smaples in datasets

        Returns datasets in order (train, valid, test)

        '''
        len_all = self.len
        test_split_size = int(np.floor(test_size * len_all)) if test_size else 0
        val_split_size = int(np.floor(val_size * len_all)) if val_size else 0
        train_split_size = len_all - test_split_size - val_split_size

        return data_utils.random_split(self, [train_split_size, val_split_size, test_split_size])
