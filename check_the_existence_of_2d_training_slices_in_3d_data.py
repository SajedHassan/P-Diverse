import os
import sys
import h5py
import numpy as np
import nibabel as nib
import torch
from collections import defaultdict

training = '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/training/'
training_2d = '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/training_2d/'

all_slices_names = sorted(os.listdir(training_2d), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])
grouped_slices_names = defaultdict(list)

# Group files by the sample number
for slices_name in all_slices_names:
    sample_number = slices_name.split("_slice")[0]  # Extract 'Sample_10', 'Sample_20', etc.
    grouped_slices_names[sample_number].append(slices_name)

# Convert the defaultdict to a regular dict (optional)
grouped_slices_names = dict(grouped_slices_names)

print(len(all_slices_names))

sample_index = 1

for sample, slices_names in grouped_slices_names.items():
    sample_t1_slices = []
    sample_t1c_slices = []
    sample_t2_slices = []

    for slice_name in slices_names:
        slice = h5py.File(os.path.join(training_2d, slice_name), 'r')

        slice_t1 = np.array(slice['t1'])
        slice_t1c = np.array(slice['t1c'])
        slice_t2 = np.array(slice['t2'])

        sample_t1_slices.append(slice_t1)
        sample_t1c_slices.append(slice_t1c)
        sample_t2_slices.append(slice_t2)

    sample_t1_slices = np.stack(sample_t1_slices, 0)
    sample_t1c_slices = np.stack(sample_t1c_slices, 0)
    sample_t2_slices = np.stack(sample_t2_slices, 0)

    sample_h5_file = h5py.File(training + sample + '.h5', 'r')
    sample_t1 = np.array(sample_h5_file['t1'])
    sample_t1c = np.array(sample_h5_file['t1c'])
    sample_t2 = np.array(sample_h5_file['t2'])

    if (sample_t1_slices == sample_t1).all() and (sample_t1c_slices == sample_t1c).all() and (sample_t2_slices == sample_t2).all():
        print(sample)
