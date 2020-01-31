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
from torch.utils.data import random_split

def get_class_dirs(glob_pattern):
    return glob.glob(glob_pattern)

def get_class_label_map(glob_pattern):
    class_dirs = glob.glob(glob_pattern)
    return {i: (class_dirs[i].split('/')[-1], class_dirs[i]) for i in range(len(class_dirs))}

def get_dataset_info(glob_pattern):
    class_dirs = glob.glob(glob_pattern)
    return {i: (class_dirs[i].split('/')[-1], class_dirs[i], tuple(glob.glob(class_dirs[i] + '/*' ))) for i in range(len(class_dirs))}

def get_files_label_map(dataset_info):
    files_label_map = []
    for class_label, class_data in dataset_info.items():
        for filename in class_data[2]:
            files_label_map.append((class_label, filename))
    return tuple(files_label_map)


def prepare_loaders(dataset_files, train_test_split_ratio=0.2, train_valid_split_ratio=0.4):
    dataset_size = len(dataset_files)
    train_subset_size = valid_train_ratio * dataset_size
    validation_subset_size = valid_train_ratio * (1 - valid_train_ratio)

    indices = list(range(dataset_size))
    validation_indices = np.random.choice(indices, size=validation_subset_size, replace=False)
    train_indices = list(set(indices) - set(validation_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    
    dataset_sizes = {
            'train': len(train_indices),
            'validation': len(validation_indices)
        }

    train_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1, sampler=train_sampler)
    validation_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1, sampler=validation_sampler)

    loaders = {
            'train': train_loader,
            'validation': validation_loader
        }

    return loaders, dataset_sizes


class MinimalUniversalDataset(Dataset):

    def __init__(self, files_label_map, transforms=None, img_resize_sizes=(64, 64)):
        #super(MinimalUniversalDataset, self).__init__()
        self.files_label_map = files_label_map
        self.dataset_len = len(files_label_map)
        self.img_resize_sizes = img_resize_sizes
        if transforms:
            self.transforms = transforms

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        class_label, path_to_image_file = self.files_label_map[idx]
        image = torch.from_numpy(img_io.imread(path_to_image_file))
        if self.img_resize_sizes:
            image = resize(image, self.img_resize_sizes)
        #print("__getitem__, image.shape: ", image.shape)
        torch_tensor = torch.from_numpy(image)
        #return image, label
        return (torch_tensor.T).permute([0, 2, 1]), class_label

def make_loaders(
        files_label_map,
        train_test_split_ratio=0.2,
        train_valid_split_ratio=0.4,
        batch_sizes=(4, 4, 4),
        img_resize_sizes=(64, 64)
    ):

    master_dataset_len = len(files_label_map)
    test_subset_len = int(master_dataset_len * train_test_split_ratio)
    train_subset_len = master_dataset_len - test_subset_len
    valid_subset_len = int(train_subset_len * train_valid_split_ratio)
    train_subset_len = train_subset_len - valid_subset_len

    train_fl_map, valid_fl_map, test_fl_map = random_split(files_label_map, [train_subset_len, valid_subset_len, test_subset_len])

    train_ds = MinimalUniversalDataset(train_fl_map, img_resize_sizes=(64, 64))
    valid_ds = MinimalUniversalDataset(valid_fl_map, img_resize_sizes=(64, 64))
    test_ds = MinimalUniversalDataset(test_fl_map, img_resize_sizes=(64, 64))

    train_loader_params = {'batch_size': batch_sizes[0], 'shuffle': True}
    valid_loader_params = {'batch_size': batch_sizes[1], 'shuffle': True}
    test_loader_params = {'batch_size': batch_sizes[2], 'shuffle': True}

    train_loader = data.DataLoader(train_ds, **train_loader_params)
    valid_loader = data.DataLoader(valid_ds, **valid_loader_params)
    test_loader = data.DataLoader(test_ds, **test_loader_params)


    return train_loader, valid_loader, test_loader


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
    '''
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

    '''

    dataset_info = get_dataset_info('../input/natural-images/natural_images/*')
    print("len(dataset_info):", len(dataset_info))
    print("type(dataset_info[0][2]): ", type(dataset_info[0][2]))
    print("dataset_info[0][2][:10]: ", dataset_info[0][2][:10])
    files_label_map = get_files_label_map(dataset_info)
    print(get_files_label_map(dataset_info)[2000:2020])
    #print(type(make_loaders(files_label_map)))
    train_dl, valid_dl, test_dl = make_loaders(files_label_map)
    print(type(train_dl), type(valid_dl), type(test_dl))
    img, label = next(iter(train_dl))
    print(img.shape, label)

    '''
    print("len(train), len(valid), len(test):", len(train), len(valid), len(test))
    train_set = set(train)
    test_set = set(test)
    valid_set = set(valid)
    print(train_set.intersection(test_set))
    print(train_set.intersection(valid_set))
    print(valid_set.intersection(test_set))
    '''
