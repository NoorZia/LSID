from os.path import join

import numpy as np
import rawpy
from skimage import color, exposure
from matplotlib import pyplot as plt

data_path = r'E:\LSID\dataset'
file_path = "Sony_test_list_reduced.txt"
files = open(join(data_path, file_path), 'r').readlines()
for i, f in enumerate(files):
    if i == 200:
        file_path_long = f.split(' ')[1]
        break

file_path_long = join(data_path, file_path_long)

gt_image = rawpy.imread(file_path_long)
gt_image = gt_image.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

# divide by 65535 after converting a uint16 intensity image to double
target = np.float32(gt_image / 65535.0)  # RGB image shape (2848, 4256, 3), channel order: RGB


# Convert image to HSV color space
img_hsv = color.rgb2hsv(target)

# Histogram normalization on V channel
img_v = img_hsv[:, :, 2].copy()
img_hsv[:, :, 2] = exposure.equalize_hist(img_hsv[:, :, 2])

# Convert back to RGB space
img_equalized = color.hsv2rgb(img_hsv.astype('float'))

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(target)
plt.subplot(2, 2, 2)
plt.imshow(img_equalized)
plt.subplot(2, 2, 3)
plt.hist(img_v.ravel())
plt.subplot(2, 2, 4)
plt.hist(img_hsv[:,:,2].ravel())
plt.show()