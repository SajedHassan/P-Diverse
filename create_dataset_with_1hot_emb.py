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
target_dataset_images_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_raw/Dataset013_NPC-1hot-emb/imagesTr/'
target_dataset_labels_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_raw/Dataset013_NPC-1hot-emb/labelsTr/'

index = 1

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

        hot_1_emb = np.ones(t1.shape)
        hot_0_emb = np.zeros(t1.shape)

        a0_hot_vec = hot_1_emb if (annotator_idx == 0) else hot_0_emb
        a1_hot_vec = hot_1_emb if (annotator_idx == 1) else hot_0_emb
        a2_hot_vec = hot_1_emb if (annotator_idx == 2) else hot_0_emb
        a3_hot_vec = hot_1_emb if (annotator_idx == 3) else hot_0_emb

        label = np.array(h5_file['label'])

        affine = np.eye(4)
        nii_t1 = nib.Nifti1Image(t1, affine)
        nii_t1c = nib.Nifti1Image(t1c, affine)
        nii_t2 = nib.Nifti1Image(t2, affine)
        nii_a0_hot_vec = nib.Nifti1Image(a0_hot_vec, affine)
        nii_a1_hot_vec = nib.Nifti1Image(a1_hot_vec, affine)
        nii_a2_hot_vec = nib.Nifti1Image(a2_hot_vec, affine)
        nii_a3_hot_vec = nib.Nifti1Image(a3_hot_vec, affine)
        nii_label = nib.Nifti1Image(label, affine)

        nib.save(nii_t1, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0000.nii.gz'))
        nib.save(nii_t1c, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0001.nii.gz'))
        nib.save(nii_t2, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0002.nii.gz'))

        nib.save(nii_a0_hot_vec, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0003.nii.gz'))
        nib.save(nii_a1_hot_vec, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0004.nii.gz'))
        nib.save(nii_a2_hot_vec, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0005.nii.gz'))
        nib.save(nii_a3_hot_vec, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0006.nii.gz'))

        nib.save(nii_label, os.path.join(target_dataset_labels_folder, new_sample_base_path + '.nii.gz'))

        h5_file.close()
        print('Annotator[', annotator_idx, '] - Saved: ', index)
        index += 1
