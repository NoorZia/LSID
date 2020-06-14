import torch
from dataset_eval import LSID
from LSID_model import LSIDModel
import os
import numpy as np
import time
import scipy
from PIL import Image
import cv2
import math
from skimage.measure import compare_ssim as ssim


def calculate_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

results_dir = './eval_results/'
model = LSIDModel()
model = model.cuda()
model.load_state_dict(torch.load(
    '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/nzia/LSID/model_checkpoints_2000/epoch-348.pt'))

model.eval()
print("Model loaded")

train_data = LSID(data_path=r"/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/nzia/LSID/dataset",
                  subset="valid", patch_size=1024)
data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=1, shuffle=None)


t0 = time.time()
psnr = []
ssimm = []
for i, (inputs, targets, name) in enumerate(data_loader):
    #print(name)

    targets = targets.permute(0, 3, 1, 2).cuda()
    inputs = inputs.permute(0, 3, 1, 2).cuda()

    outputs = model(inputs)

    output_image = outputs.permute(0,2,3,1).cpu().data.numpy()
    target_image = targets.permute(0,2,3,1).cpu().data.numpy()
    target_image = target_image[0,:,:,:]
    output_image = output_image[0,:,:,:]
    
    psnr.append(calculate_psnr(target_image, output_image))
    ssimm.append(ssim(target_image, output_image, multichannel=True))


print("PSNR ", sum(psnr)/len(psnr))
print("SSIM ", sum(ssimm)/len(ssimm))

  