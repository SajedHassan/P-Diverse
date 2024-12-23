import h5py
import os
import numpy as np
import nibabel as nib

base_dataset_folders = [
    '/home/sajed/thesis/MMIS/dataset/splitted/training_2d/a0/',
    '/home/sajed/thesis/MMIS/dataset/splitted/training_2d/a1/',
    '/home/sajed/thesis/MMIS/dataset/splitted/training_2d/a2/',
    '/home/sajed/thesis/MMIS/dataset/splitted/training_2d/a3/'
]
target_dataset_images_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_raw/Dataset013_NPC-special_mask/imagesTr/'
target_dataset_labels_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_raw/Dataset013_NPC-special_mask/labelsTr/'

index = 1

special_masks = [0.2, 0.4, 0.6, 0.8]

for annotator_idx, base_dataset_folder in enumerate(base_dataset_folders):
    sorted_files_list = sorted(os.listdir(base_dataset_folder), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])
    for file_name in sorted_files_list:
        if not file_name.endswith('.h5'):
            continue
        file_path = os.path.join(base_dataset_folder, file_name)
        file_name_parts = file_name.replace('.', '_').split("_")
        new_sample_base_path = 'NPC' + file_name_parts[1] + '-' + file_name_parts[3] + '_' + "{:03}".format(index)

        h5_file = h5py.File(file_path, 'r')
        t1 = np.array(h5_file['t1'])
        t1c = np.array(h5_file['t1c'])
        t2 = np.array(h5_file['t2'])

        special_mask = np.full(t1.shape, special_masks[annotator_idx])

        label = np.array(h5_file['label'])

        affine = np.eye(4)
        nii_t1 = nib.Nifti1Image(t1, affine)
        nii_t1c = nib.Nifti1Image(t1c, affine)
        nii_t2 = nib.Nifti1Image(t2, affine)
        nii_special_mask = nib.Nifti1Image(special_mask, affine)
        nii_label = nib.Nifti1Image(label, affine)

        nib.save(nii_t1, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0000.nii.gz'))
        nib.save(nii_t1c, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0001.nii.gz'))
        nib.save(nii_t2, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0002.nii.gz'))

        nib.save(nii_special_mask, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0003.nii.gz'))

        nib.save(nii_label, os.path.join(target_dataset_labels_folder, new_sample_base_path + '.nii.gz'))

        h5_file.close()
        print('Annotator[', annotator_idx, '] - Saved: ', index)
        index += 1
