import numpy as np
import rawpy
from skimage import color, exposure
from matplotlib import pyplot as plt

file_path_long = r'E:\LSID\dataset\Sony\long\00100_00_30s.ARW'

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
plt.title('Unmodified')
plt.imshow(target)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Histogram equalised')
plt.imshow(img_equalized)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.hist(img_v.ravel(), bins=256)
plt.xlabel('Pixel intensity in HSV value channel')
plt.ylabel('Number of pixels')
ax = plt.gca()
plt.gca().set_yticklabels([])

plt.subplot(2, 2, 4, sharey=ax)
plt.hist(img_hsv[:, :, 2].ravel(), bins=256)
plt.xlabel('Pixel intensity in HSV value channel')
plt.ylabel('Number of pixels')
plt.gca().set_yticklabels([])

plt.show()
