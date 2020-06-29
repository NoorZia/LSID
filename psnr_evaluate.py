import torch
from dataset_eval import LSID
from LSID_model import LSIDModel
from LSID_plus_model import LSIDplusModel
from lsid_update import LSIDplusModelconv
import os
import numpy as np
import time
import scipy
from PIL import Image
import cv2
import math
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as pssnr


train_data = LSID(data_path=r"/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/nzia/LSID/dataset",
                  subset="valid")
data_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=None)

model = LSIDModel()
model = model.cuda()
model.load_state_dict(torch.load(
    '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/nzia/LSID/models/model_reduced_500/epoch-499.pt'))
model = torch.nn.DataParallel(model)
model.eval()
print("Model loaded")

t0 = time.time()
psnr = []
ssimm = []
times = []

with torch.no_grad():
    for i, (inputs, targets, name) in enumerate(data_loader):
 
        targets = targets.permute(0, 3, 1, 2).cuda()
        inputs = inputs.permute(0, 3, 1, 2).cuda()

        t0 = time.time()
        outputs = model(inputs)
        times.append(time.time()-t0)

        output_image = outputs.permute(0,2,3,1).cpu().data.numpy()
        target_image = targets.permute(0,2,3,1).cpu().data.numpy()

        target_image = target_image[0,:,:,:]
        output_image = output_image[0,:,:,:]
      
        p_ = pssnr(target_image, output_image)
        s_ = ssim(target_image, output_image, multichannel=True)
        
        psnr.append(p_)
        ssimm.append(s_)


print("PSNR unet full", sum(psnr)/len(psnr))
print("SSIM unet full", sum(ssimm)/len(ssimm))

print("Average time", sum(times)/len(times))

