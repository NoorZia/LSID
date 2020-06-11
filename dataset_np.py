import numpy as np
import torch
import rawpy
from torch.utils.data import Dataset
import random
from PIL import Image
from scipy import ndimage
from os.path import join
import time
import os

patch_size = 512


class LSID(Dataset):
    def __init__(self, data_path, subset, patch_size=512):
        self.data_path = data_path
        self.gt_images = {}
        self.train_images = {}
        self.patch_size = patch_size
        self.data = self.__make_dataset(subset)
        
        

    def __pack_bayer(self, image):
        # change raw image to float
        image = image.raw_image_visible.astype(np.float32)
       
        image = np.maximum(image - 512, 0) / (16383 - 512)
        image = np.expand_dims(image, axis=2)  # (x,x,dim)
        h, w, _ = image.shape
        return np.concatenate((image[0:h:2, 0:w:2], image[0:h:2, 1:w:2], image[1:h:2, 0:w:2], image[1:h:2, 1:w:2]),
                              axis=2)  # concat along dim (x,x,dim)

    def __make_dataset(self, subset):
        file_path = "Sony_train_list_npy.txt"
        if subset == 'train':
            file_path = "Sony_train_list_npy.txt"
        elif subset == "test":
            file_path = "Sony_test_list_npy.txt"
        elif subset == "valid":
            file_path = "Sony_val_list_npy.txt"
        files = open(join(self.data_path, file_path), 'r').readlines()

        for img in os.listdir("dataset/Sony/long"):
            image = rawpy.imread("dataset/Sony/long/" + img)
            image = image.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            target = np.float32(image / 65535.0)  # divide by 65535 after converting a uint16 intensity image to double
            self.gt_images["dataset/Sony/long/" + img] = target

        for img in os.listdir("dataset/Sony/short"):
            image = rawpy.imread("dataset/Sony/short/" + img)
            image = self.__pack_bayer(image)
            self.train_images["dataset/Sony/short/" + img] = image


        

        dataset = []
        for f in files:  # TODO: change back!!
            
            file_list = f.split()
            file_path_short = join('dataset/Sony/short/', os.path.basename(os.path.splitext(file_list[0])[0]) + '.ARW')
            #file_path_long = join(self.data_path, file_list[1])
            
            file_path_long = join('dataset/Sony/long/', os.path.basename(os.path.splitext(file_list[1])[0]) + '.ARW')
            
            
            exposure_ratio = float(file_list[1].split("_")[-1][:-5]) / float(file_list[0].split("_")[-1][:-5])
            iso = file_list[2]
            #target = np.load(file_path_long)
            
            sample = {
                'image': file_path_short,
                'gt': file_path_long,
                'exposure_ratio': exposure_ratio,
                'iso': iso
            }
            dataset.append(sample)

        return dataset

    def __getitem__(self, index):
 
        file_path_short = self.data[index]['image']
        file_path_long = self.data[index]['gt']

        image =  self.train_images[file_path_short]
        image = image*min(self.data[index]['exposure_ratio'], 300)

        target = self.gt_images[file_path_long]
        


        # random crop
        i, j = random.randint(0, image.shape[0] - self.patch_size), random.randint(0, image.shape[1] - self.patch_size)

        image = image[i:i + self.patch_size, j:j + self.patch_size, :]
        target = target[i * 2:i * 2 + self.patch_size * 2, j * 2:j * 2 + self.patch_size * 2, :]

        # changed from tensor functions to numpy functions as tensor need to converted to PIL image to use built in transforms

        # random rotation
        """if random.random() > 0.5:
            angle = random.randint(-10, 10)
            image = ndimage.rotate(image, angle, reshape=False)  # set to false to preserve size
            target = ndimage.rotate(target, angle, reshape=False)"""

        image = torch.from_numpy(image)
        target = torch.from_numpy(target)

        # flip with tensor type to avoid negative strides

        if random.random() > 0.5:
            image = torch.flip(image, dims=(0,))
            target = torch.flip(target, dims=(0,))

        if random.random() > 0.5:
            image = torch.flip(image, dims=(1,))
            target = torch.flip(target, dims=(1,))
        

        return image, target

    def __len__(self):
        return len(self.data)


#train_data = LSID("./", "train")
#data_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=None)

#for i, (inputs, targets) in enumerate(data_loader):
#    print(i, inputs.shape, targets.shape)




