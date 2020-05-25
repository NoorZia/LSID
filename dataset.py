
import numpy as np 
import torch
import rawpy
from torch.utils.data import Dataset
import random
from PIL import Image
from scipy import ndimage

patch_size = 512

class LSID(Dataset):
    def __init__(self, root_path, subset, patch_size = 512):
        self.data = self.__make_dataset(root_path,  subset)
        self.patch_size = patch_size

    def __pack_bayer(self, image):
        #change raw image to float
        image = image.raw_image_visible.astype(np.float32)
        image = np.maximum(image - 512,0)/(16383 - 512)
        image = np.expand_dims(image, axis = 2) #(x,x,dim) 
        h,w,_ = image.shape
        return np.concatenate((image[0:h:2, 0:w:2], image[0:h:2, 1:w:2], image[1:h:2, 0:w:2], image[1:h:2, 1:w:2]), axis = 2) # concat along dim (x,x,dim)

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
            long_image = long_image.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            long_image = np.float32(long_image / 65535.0) #divide by 65535 after converting a uint16 intensity image to double


            short_image = rawpy.imread(file_list[0])
            short_image = self.__pack_bayer(short_image)
            
            exposure = float(file_list[1].split("_")[-1][:-5])/float(file_list[0].split("_")[-1][:-5])
            
            short_image = short_image*min(exposure, 300)



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
    
        i = random.randint(0, image.shape[0] - self.patch_size)
        j = random.randint(0, image.shape[1] - self.patch_size)
        i,j = random.randint(0, image.shape[0] - self.patch_size), random.randint(0, image.shape[1] - self.patch_size)
    
        print(i,j, self.patch_size)
        image = image[i:i+self.patch_size, j:j+self.patch_size, :]
        target = target[ i*2:i*2 + self.patch_size*2, j*2:j*2 + self.patch_size*2, :]


        # changed from tensor functions to numpy functions as tensor need to converted to PIL image to use built in transforms

        
        # random rotation
        if random.random() > 0.5:
            angle = random.randint(-10,10)
            image = ndimage.rotate(image,angle, reshape=False) # set to false to preserve size
            target = ndimage.rotate(target, angle, reshape=False)
        print("image",image.shape, "target",target.shape)

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

train_data = LSID("./", "train")
data_loader = torch.utils.data.DataLoader(train_data, batch_size = 2, shuffle = None )

for i, (inputs, targets) in enumerate(data_loader):
    print(i, inputs.shape, targets.shape)




