import torch
from dataset import LSID
from LSID_model import LSIDModel
import os
import numpy as np
import time
import scipy
from PIL import Image



results_dir = './eval_results/'

model = LSIDModel()
model = model.cuda()
model.load_state_dict(torch.load(
    'models/epoch-349.pt'))
model = torch.nn.DataParallel(model)
model.eval()
print("Model loaded")

train_data = LSID(data_path=r"dataset",
                  subset="valid", patch_size=1024)

data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=1, shuffle=None)

with torch.no_grad():

    for i, (inputs, targets, name) in enumerate(data_loader):
        
        inputs = inputs.permute(0, 3, 1, 2).cuda()
        outputs = model(inputs)
        output_image = outputs.permute(0,2,3,1).cpu().data.numpy()
        
        scipy.misc.toimage(output_image[0,:,:,:]*255,  high=255, low=0, cmin=0, cmax=255).save(results_dir + name[0] + '.jpg')

    

