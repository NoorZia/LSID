import numpy as np
import matplotlib.pyplot as plt
import os

# SETTINGS HIGHLY SPECIFIC FOR MY DATA & COMPUTER!! ==============
data_path = r'E:\LSID\results\fuji200'
file_list_path = os.path.join(data_path, 'Fuji_val_list.txt')

# Load PSNR and SSIM values from the evaluation experiments.
# The values occur in the same order as in the
# Fuji_val_list.txt file because of how I saved them.
psnr = np.load(os.path.join(data_path, 'PSNR.npy'))
ssim = np.load(os.path.join(data_path, 'SSIM.npy'))
# ================================================================

# Read the file list used in the evaluation experiment
files = open(file_list_path, 'r').readlines()

# Empty numpy arrays
exposure_time_arr = np.zeros(shape=len(files))
iso_arr = np.zeros(shape=len(files))
f_number_arr = np.zeros(shape=len(files))

for i, f in enumerate(files):
    f_list = f.split()
    # Example: f_list == ['./Fuji/short/20015_00_0.1s.RAF', './Fuji/long/20015_00_10s.RAF', 'ISO1000', 'F2.8']
    exposure_time = float(f_list[0].split('_')[-1][:-5])
    iso = float(f_list[2][3:])
    f_number = float(f_list[-1][1:])

    exposure_time_arr[i] = exposure_time
    iso_arr[i] = iso
    f_number_arr[i] = f_number

plt.figure(figsize=(15, 10))
plt.suptitle('Fuji, 200 epochs, RGB input', fontweight="bold")

plt.subplots_adjust(hspace=0.3)

plt.subplot(2, 2, 1)
plt.title('PSNR coloured by ISO')
plt.xlabel('Exposure_time [s]')
plt.ylabel('PSNR')
plt.scatter(exposure_time_arr, psnr, s=5, c=iso_arr, cmap='cool')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title('SSIM coloured by ISO')
plt.ylabel('SSIM')
plt.xlabel('Exposure_time [s]')
plt.scatter(exposure_time_arr, ssim, s=5, c=iso_arr, cmap='cool')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title('PSNR coloured by exposure_time [s]')
plt.xlabel('ISO')
plt.scatter(iso_arr, psnr, s=5, c=exposure_time_arr, cmap='winter')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title('SSIM coloured by exposure_time [s]')
plt.ylabel('SSIM')
plt.xlabel('ISO')
plt.scatter(iso_arr, ssim, s=5, c=exposure_time_arr, cmap='winter')
plt.colorbar()

plt.show()
