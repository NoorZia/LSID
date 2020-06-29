import torch
from dataset import LSID
import os
import numpy as np
import time
import scipy
from PIL import Image

num_epochs = 500
model_dir = 'models/'
results_dir = 'results/'


model = LSIDModel()
model = model.cuda()

train_data = LSID(data_path=r"dataset", subset="train")
data_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=None)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.L1Loss()

t0 = time.time()
for epoch in range(0,num_epochs):
    for i, (inputs, targets, name) in enumerate(data_loader):
 
        targets = targets.permute(0,3,1,2).cuda() 
        inputs = inputs.permute(0,3,1,2).cuda() 

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        print("Epoch:", epoch, "iteration:", i, "Loss", loss.data)
       

    torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
    output_image = outputs.permute(0,2,3,1).cpu().data.numpy()
    target_image = targets.permute(0,2,3,1).cpu().data.numpy()
    result = np.concatenate(( target_image[0,:,:,:],output_image[0,:,:,:]),axis=1)
    #print(result.shape)
    scipy.misc.toimage(result*255,  high=255, low=0, cmin=0, cmax=255).save(results_dir + '%04d_%05d_00_train.jpg'%(epoch,i))

   
