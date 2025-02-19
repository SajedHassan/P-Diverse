import h5py
import os
import numpy as np
import nibabel as nib

base_dataset_folders = [
    '/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_1/val/'
]
target_dataset_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_input/Dataset024_LIDC-learnable_emb_spade_enc_dec_FOLD_1_validation_data/'

index = 1

for _, base_dataset_folder in enumerate(base_dataset_folders):
    sorted_files_list = sorted(os.listdir(base_dataset_folder), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])
    for file_name in sorted_files_list:
        for generated_AnnotatorIdx in range(0, 4):
            if not file_name.endswith('.h5'):
                continue
            file_path = os.path.join(base_dataset_folder, file_name)
            file_name_parts = file_name.replace('.', '_').split("_")
            new_sample_base_path = 'LIDC-' + file_name_parts[0] + '-' + file_name_parts[1] + '-' + str(generated_AnnotatorIdx) + '_' + "{:03}".format(index)

            h5_file = h5py.File(file_path, 'r')
            ct = np.array(h5_file['image'])

            affine = np.eye(4)
            nii_ct = nib.Nifti1Image(ct, affine)

            nib.save(nii_ct, os.path.join(target_dataset_folder, new_sample_base_path + '_0000.nii.gz'))
            with open(os.path.join(target_dataset_folder, new_sample_base_path + '_type.txt'), "w") as txt_file:
                txt_file.write(str(generated_AnnotatorIdx))

            h5_file.close()
            print('Annotator[', generated_AnnotatorIdx, '] - Saved: ', index)
        index += 1
