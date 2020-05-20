
import os
import numpy as np 
import torch
import rawpy
import glob
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import random

patch_size = 512

class LSID(Dataset):
    def __init__(self, root_path, subset):
        self.data = self.__make_dataset(root_path,  subset)

    def __make_dataset(self, root_path, subset):
        file_path = "dataset/Sony_train_list.txt"
        if subset == 'train':
            file_path = "dataset/Sony_train_list.txt"
        elif subset == "test":
            file_path = "dataset/Sony_test_list.txt"
        elif subset == "valid":
            file_path = "dataset/Sony_val_list.txt"
        files = open(file_path, 'r').readlines()
        
        dataset = []
        for f in files:
            file_list = f.split()
            long_image = rawpy.imread(file_list[1])
            short_image = rawpy.imread(file_list[0])
            iso = file_list[2]
            exposure = file_list[3]

            sample = {
                'image': short_image,
                'gt' : long_image,
                'f' : exposure,
                'iso' : iso
            }
            dataset.append(sample)

        return dataset
    
    def __getitem__(self, index):
        image = self.data[index]['image']
        target =  self.data[index]['gt']

        # random crop
        i,j,h,w = transforms.RandomCrop.get_params(image, output_size=(512,512))
        image = TF.crop(image, i,j,h,w)
        target = TF.crop(target, i, j, h, w)

        # random flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)
        
        if random.random() > 0.5:
            image = TF.vflip(image)
            target = TF.vflip(target)
        
        # random rotation
        if random.random() > 0.5:
            angle = random.randint(-10,10)
            image = TF.rotate(image, angle)
            target = TF.rotate(target, angle)
        
        return image, target







