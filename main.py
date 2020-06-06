import torch
from dataset import LSID
from LSID_model import LSIDModel
import os
import numpy as np
import time
import scipy
from PIL import Image

num_epochs = 1
model_dir = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/nzia/LSID/model_checkpoints/'
results_dir = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/nzia/LSID/result_images/'


model = LSIDModel()
model = model.cuda()

train_data = LSID(data_path=r"/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/nzia/LSID/dataset", subset="train")
data_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=None)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.L1Loss()

t0 = time.time()
for epoch in range(num_epochs):


    #print(os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
    for i, (inputs, targets) in enumerate(data_loader):

        print("Epoch:", epoch, "iteration:", i)
        # TODO: check dimension ordering in dataloader: currently [B, H, W, C], expected to be [B, C, H, W]
        targets = targets.permute(0,3,1,2).cuda() 
        inputs = inputs.permute(0,3,1,2).cuda() # permuting here instead of in data loader
        
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        print("Loss", loss)
        loss.backward()
        optimizer.step()
       

    torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
    output_image = outputs.permute(0,2,3,1).cpu().data.numpy()
    target_image = targets.permute(0,2,3,1).cpu().data.numpy()
    result = np.concatenate(( target_image[0,:,:,:],output_image[0,:,:,:]),axis=1)
    print(result.shape)
    scipy.misc.toimage(result*255,  high=255, low=0, cmin=0, cmax=255).save(results_dir + '%04d_%05d_00_train.jpg'%(epoch,i))

   