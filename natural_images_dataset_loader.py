import glob
import os

import numpy as np
from PIL import Image
import skimage
from skimage import io as img_io
from skimage import color
from skimage.transform import rescale, resize, downscale_local_mean

import torch
from torch.utils import data
from torch.utils.data import Dataset

def get_class_dirs(glob_pattern):
    return glob.glob(glob_pattern)

def get_class_label_map(glob_pattern):
    class_dirs = glob.glob(glob_pattern)
    return {i: (class_dirs[i].split('/')[-1], class_dirs[i]) for i in range(len(class_dirs))}


class NaturalImagesDataset(Dataset):

    def __init__(self, class_label_map, valid_ratio= 1/4, valid=False, resize_sizes=(64, 64)):
        self.class_label_map = class_label_map
        self.valid_ratio = valid_ratio
        self.valid = valid
        self.train_filenames_num = 0
        self.valid_filenames_num = 0
        self.resize_sizes = resize_sizes
        self.filenames_label_map = self.__get_filenames_label_map__()
    
    def __len__(self):
        self.dataset_len = len(glob.glob('../input/natural-images/natural_images/*' + '*/*'))
        if self.valid:
            return self.valid_filenames_num
        else:
            return self.train_filenames_num

    def __getitem__(self, idx):
        path_to_image_file, label = self.filenames_label_map[idx]
        image = torch.from_numpy(img_io.imread(path_to_image_file))
        if self.resize_sizes:
            image = resize(image, self.resize_sizes)
        #print("__getitem__, image.shape: ", image.shape)
        torch_tensor = torch.from_numpy(image)
        #return image, label
        return (torch_tensor.T).permute([0, 2, 1]), label

    def __get_filenames_label_map__(self):
        train_label_filenames_map = []
        valid_label_filenames_map = []
        for label, class_description in self.class_label_map.items():
            filenames = glob.glob(class_description[1] + '/*')
            filenames_num = len(filenames)
            valid_filenames_num = int(filenames_num * self.valid_ratio)
            train_filenames_num = filenames_num - valid_filenames_num

            self.train_filenames_num += train_filenames_num
            self.valid_filenames_num += valid_filenames_num

            for filename in filenames[:train_filenames_num]:
                train_label_filenames_map.append((filename, label))
            for filename in filenames[train_filenames_num:]:
                 valid_label_filenames_map.append((filename, label))
        if self.valid:
            self.filenames_label_map = valid_label_filenames_map
        else:
            self.filenames_label_map = train_label_filenames_map

        return self.filenames_label_map

    
if __name__ == '__main__':
    params = {'batch_size': 16, 'shuffle': True}
    #'../input/natural-images/natural_images/*'
    class_label_map = get_class_label_map('../input/natural-images/natural_images/*')
    #ni_ds = NaturalImagesDataset('../input/natural-images/natural_images/*')
    ni_ds = NaturalImagesDataset(class_label_map)
    #print(len(ni_ds))
    #print(ni_ds[10])
    #print(glob.glob('../input/natural-images/natural-images/*'))
    #print(ni_ds.__get_filenames_label_map__()[:10])
    ni_gen = data.DataLoader(ni_ds, **params)
    #print(dir(ni_gen))

    for batch, label in ni_gen:
        print("type(batch): ", type(batch))
        print("batch.shape:", batch.shape)
        print("label: ", label)
