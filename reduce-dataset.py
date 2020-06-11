""" Script for reducing datasets. Produces new text files """
from os.path import join
from collections import defaultdict

# Settings
data_path = "E:\LSID\dataset"
nr_short_per_gt = 6  # number of images per ground truth in the reduced set(s)


def reduce_dataset(subset='train', nr_short_per_gt=6):
    file_path = "Sony_{}_list.txt".format(subset)  # The text file to be reduced
    file_path_reduced = join(data_path, file_path)[:-4] + '_reduced.txt'  # where to save the new text file

    files = open(join(data_path, file_path), 'r').readlines()

    gt_id_previous = None
    lines = []  # List to fill with lines corresponding to a specific ground truth image

    with open(file_path_reduced, 'w') as new_txt_file:
        for f in files:
            file_list = f.split()
            file_name_short = file_list[0]  # Example: './Sony/short/00001_06_0.1s.ARW'

            gt_id = int(file_name_short.split(sep='_')[0][-5:])  # Example: 00001

            # If we looped over all images corresponding to one gt (they come in order)
            if gt_id != gt_id_previous and gt_id_previous is not None:
                chosen_lines = lines
                lines = []

                chosen_lines.sort()  # This sorts them after burst-number effectively
                chosen_lines = chosen_lines[0:nr_short_per_gt]  # Choose the nr_short_per_gt first lines (= short imgs)
                chosen_lines_string = ''.join(chosen_lines)

                # Write the chosen lines to the new text file
                new_txt_file.write(chosen_lines_string)

            lines.append(f)
            gt_id_previous = gt_id

    # This is just for double check that the number of images are correct, and see how many of each shutter speed
    # we have
    files_reduced = open(file_path_reduced, 'r').readlines()
    gt_images = set()
    nr_short_images = 0
    dd = defaultdict(int)
    for f in files_reduced:
        file_list = f.split()
        nr_short_images += 1
        gt_id = int(file_list[1].split(sep='_')[0][-5:])
        gt_images.add(gt_id)
        shutter = file_list[0].split("_")[-1][:-5]
        dd[shutter] += 1

    return len(gt_images), nr_short_images, dd


def main():
    for subset in ['train', 'val', 'test']:
        # Reduce dataset and create txt file
        nr_gt, nr_short, nr_per_shutter = reduce_dataset(subset=subset, nr_short_per_gt=nr_short_per_gt)

        # Print some stats
        print('==== {}_reduced ===='.format(subset))
        print('Total nr gt:', nr_gt)
        print('Total nr short:', nr_short)
        print('nr images per shutter speed:', str(nr_per_shutter))


if __name__ == '__main__':
    main()
