""" Script for evaluating the dataset reduction condition """

from os.path import join
import matplotlib.pyplot as plt

data_path = "E:\dark-data\LSID\dataset"
subset = "train"


def count_images(data_path, subset, nr_images_per_gt):
    if subset == 'train':
        file_path = "Sony_train_list.txt"
    elif subset == "test":
        file_path = "Sony_test_list.txt"
    elif subset == "valid":
        file_path = "Sony_val_list.txt"

    files = open(join(data_path, file_path), 'r').readlines()

    count_reduced = 0
    count = 0
    for f in files:
        file_list = f.split()
        image_path = file_list[0]  # Example: './Sony/short/00001_06_0.1s.ARW'
        image_number_string = image_path.split(sep='_')[1]
        image_number = int(image_number_string)

        count += 1
        if image_number > nr_images_per_gt:  # This condition can be used to reduce the set
            continue
        count_reduced += 1

    return count, count_reduced


count_ = []
count_reduced_ = []
for i in range(14):
    count, count_reduced = count_images(data_path, subset, i)
    count_.append(count)
    count_reduced_.append(count_reduced)

plt.plot(count_reduced_)
plt.title(subset)
plt.xlabel('Max number of images per ground truth image')
plt.ylabel('Number of images in {} set'.format(subset))
plt.show()

print(count_reduced_)
