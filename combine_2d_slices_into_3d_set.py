import os
import sys
import h5py
import numpy as np
import nibabel as nib
import torch
from collections import defaultdict

target_samples_3d = '/home/sajed/thesis/MMIS/dataset/GENERATED_NPC_FROM_NNUNET_WITH_LEARNABLE_EMB_E32_SM_ES_DS_F5_P250_(1)/testing_generated/'
slices_2d = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_output/E32 SM ES DS F5 P250 (1)_Dataset021_NPC-learnable_emb_spade_enc_dec_with_validation_250_epochs_testing_data_PP/'

all_slices_names = sorted(os.listdir(slices_2d), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])
grouped_slices_names = defaultdict(list)

# Group files by the sample number
for slices_name in all_slices_names:
    sample_number = slices_name.split("_")[0].split('-')[0] # Extract 'NPC0', 'NPC1', etc.
    grouped_slices_names[sample_number].append(slices_name)

# Convert the defaultdict to a regular dict (optional)
grouped_slices_names = dict(grouped_slices_names)

print(len(all_slices_names))

sample_index = 1

for sample, slices_names in grouped_slices_names.items():
    ann0_sample_seg = []
    ann1_sample_seg = []
    ann2_sample_seg = []
    ann3_sample_seg = []

    for slice_index in range(0, len(slices_names), 4):
        ann0_sample_seg.append(nib.load(os.path.join(slices_2d, slices_names[slice_index + 0])).get_fdata())
        ann1_sample_seg.append(nib.load(os.path.join(slices_2d, slices_names[slice_index + 1])).get_fdata())
        ann2_sample_seg.append(nib.load(os.path.join(slices_2d, slices_names[slice_index + 2])).get_fdata())
        ann3_sample_seg.append(nib.load(os.path.join(slices_2d, slices_names[slice_index + 3])).get_fdata())

    ann0_sample_seg = np.stack(ann0_sample_seg, 0)
    ann1_sample_seg = np.stack(ann1_sample_seg, 0)
    ann2_sample_seg = np.stack(ann2_sample_seg, 0)
    ann3_sample_seg = np.stack(ann3_sample_seg, 0)

    sample_h5_file = h5py.File(target_samples_3d + 'Sample_' + sample[3:] + '.h5', 'w')
    sample_h5_file.create_dataset('gen_label_a1', data=ann0_sample_seg)
    sample_h5_file.create_dataset('gen_label_a2', data=ann1_sample_seg)
    sample_h5_file.create_dataset('gen_label_a3', data=ann2_sample_seg)
    sample_h5_file.create_dataset('gen_label_a4', data=ann3_sample_seg)
    sample_h5_file.close()

