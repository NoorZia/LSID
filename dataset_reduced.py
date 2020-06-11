import numpy as np
import torch
import rawpy
from scipy import ndimage
from torch.utils.data import Dataset
import random
from os.path import join


class LSIDReduced(Dataset):
    def __init__(self, data_path, subset, patch_size=512):
        self.data_path = data_path
        self.patch_size = patch_size
        self.gt_images = {}  # Dictionary for ground truth images
        self.train_images = {}  # Dictionary for short exposure images
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
        """ Fill the dictionaries self.gt_images and self.train_images with data (images) """

        if subset == 'train':
            file_path = "Sony_train_list_reduced.txt"  # <--- the reduced text files
        elif subset == "test":
            file_path = "Sony_test_list_reduced.txt"
        elif subset == "val":
            file_path = "Sony_val_list_reduced.txt"

        files = open(join(self.data_path, file_path), 'r').readlines()

        dataset = []
        for f in files:
            file_list = f.split()
            # Example: ['./Sony/short/00001_00_0.04s.ARW', './Sony/long/00001_00_10s.ARW', 'ISO200', F8]

            file_path_short = join(self.data_path, file_list[0])
            file_path_long = join(self.data_path, file_list[1])

            # Load the target image
            if file_list[1] not in self.gt_images.keys():
                gt_image = rawpy.imread(file_path_long)
                gt_image = gt_image.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                # divide by 65535 after converting a uint16 intensity image to double
                target = np.float32(gt_image / 65535.0)
                self.gt_images[file_list[1]] = target

            # Load the short image
            image_short = rawpy.imread(file_path_short)
            image_short = self.__pack_bayer(image_short)
            self.train_images[file_list[0]] = image_short

            # exposure_ratio = ratio of the shutter speeds, which are specified in the file names
            exposure_ratio = float(file_list[1].split("_")[-1][:-5]) / float(file_list[0].split("_")[-1][:-5])

            # Get iso
            iso = file_list[2]

            sample = {
                'image': file_list[0],
                'gt': file_list[1],
                'exposure_ratio': exposure_ratio,
                'iso': iso
            }

            dataset.append(sample)

        return dataset

    def __getitem__(self, index):
        """ Returns short exposure and target images from the dataset - after performing random augmentation steps. """

        # Retrieve the keys for the data (to find it in the dictionaries)
        key_short = self.data[index]['image']
        key_long = self.data[index]['gt']

        # Get the short exposure image
        image = self.train_images[key_short]
        image = image * min(self.data[index]['exposure_ratio'], 300)

        # Get the target image
        target = self.gt_images[key_long]

        # Data augmentation:
        # (1) Random crop
        i, j = random.randint(0, image.shape[0] - self.patch_size), random.randint(0, image.shape[1] - self.patch_size)

        image = image[i:i + self.patch_size, j:j + self.patch_size, :]
        target = target[i * 2:i * 2 + self.patch_size * 2, j * 2:j * 2 + self.patch_size * 2, :]

        # (2) Random rotation (takes a lot of time)
        """if random.random() > 0.5:
            angle = random.randint(-10, 10)
            image = ndimage.rotate(image, angle, reshape=False)  # set to false to preserve size
            target = ndimage.rotate(target, angle, reshape=False)"""

        image = torch.from_numpy(image)
        target = torch.from_numpy(target)

        # (3) Random flip. flip with tensor type to avoid negative strides
        if random.random() > 0.5:
            image = torch.flip(image, dims=(0,))
            target = torch.flip(target, dims=(0,))

        if random.random() > 0.5:
            image = torch.flip(image, dims=(1,))
            target = torch.flip(target, dims=(1,))

        return image, target

    def __len__(self):
        """ Returns the number of (short exposure image, ground truth image) pairs in the dataset. """
        return len(self.data)
