import h5py
import os
import numpy as np
import nibabel as nib

base_dataset_folders = [
    '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/splitted/training/a0/',
    '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/splitted/training/a1/',
    '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/splitted/training/a2/',
    '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/splitted/training/a3/',
    '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/splitted/validation/a0/',
    '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/splitted/validation/a1/',
    '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/splitted/validation/a2/',
    '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/splitted/validation/a3/'
]
target_dataset_images_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_raw/Dataset022_NPC-3d_learnable_emb_spade_enc_dec_with_validation/imagesTr/'
target_dataset_labels_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_raw/Dataset022_NPC-3d_learnable_emb_spade_enc_dec_with_validation/labelsTr/'

index = 1

for annotator_idx, base_dataset_folder in enumerate(base_dataset_folders):
    annotator_idx = annotator_idx % 4
    sorted_files_list = sorted(os.listdir(base_dataset_folder), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])
    for file_name in sorted_files_list:
        if not file_name.endswith('.h5'):
            continue
        file_path = os.path.join(base_dataset_folder, file_name)
        file_name_parts = file_name.replace('.', '_').split("_")
        new_sample_base_path = 'NPC' + file_name_parts[1] + '_' + "{:03}".format(index)

        h5_file = h5py.File(file_path, 'r')
        t1 = np.array(h5_file['t1'])
        t1c = np.array(h5_file['t1c'])
        t2 = np.array(h5_file['t2'])

        try:
             label = np.array(h5_file['label'])
        except:
             label = np.array(h5_file['label_a' + str(annotator_idx + 1)])

        affine = np.eye(4)
        nii_t1 = nib.Nifti1Image(t1, affine)
        nii_t1c = nib.Nifti1Image(t1c, affine)
        nii_t2 = nib.Nifti1Image(t2, affine)
        nii_label = nib.Nifti1Image(label, affine)

        nib.save(nii_t1, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0000.nii.gz'))
        nib.save(nii_t1c, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0001.nii.gz'))
        nib.save(nii_t2, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0002.nii.gz'))
        with open(os.path.join(target_dataset_images_folder, new_sample_base_path + '_type.txt'), "w") as txt_file:
            txt_file.write(str(annotator_idx))

        nib.save(nii_label, os.path.join(target_dataset_labels_folder, new_sample_base_path + '.nii.gz'))

        h5_file.close()
        print('Annotator[', annotator_idx, '] - Saved: ', index)
        index += 1
