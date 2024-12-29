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
target_dataset_images_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_input/Dataset015_NPC-learnable_emb_10_generalized/'

index = 1

number_of_generalized_samples = 10

for annotator_idx, base_dataset_folder in enumerate(base_dataset_folders):
    sorted_files_list = sorted(os.listdir(base_dataset_folder), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])
    for file_name in sorted_files_list:
        if not file_name.endswith('.h5'):
            continue

        for generated_AnnotatorIdx in range(0, number_of_generalized_samples):
            if (annotator_idx == generated_AnnotatorIdx):
                continue # generate only the missing masks

            file_path = os.path.join(base_dataset_folder, file_name)
            file_name_parts = file_name.replace('.', '_').split("_")
            new_sample_base_path = 'NPC' + file_name_parts[1] + '-' + file_name_parts[3] + '-' + str(generated_AnnotatorIdx) + '_' + "{:03}".format(index)

            h5_file = h5py.File(file_path, 'r')
            t1 = np.array(h5_file['t1'])
            t1c = np.array(h5_file['t1c'])
            t2 = np.array(h5_file['t2'])

            affine = np.eye(4)
            nii_t1 = nib.Nifti1Image(t1, affine)
            nii_t1c = nib.Nifti1Image(t1c, affine)
            nii_t2 = nib.Nifti1Image(t2, affine)

            nib.save(nii_t1, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0000.nii.gz'))
            nib.save(nii_t1c, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0001.nii.gz'))
            nib.save(nii_t2, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0002.nii.gz'))

            with open(os.path.join(target_dataset_images_folder, new_sample_base_path + '_type.txt'), "w") as txt_file:
                txt_file.write(str(generated_AnnotatorIdx))

            h5_file.close()
            print('Annotator[', annotator_idx, '] - Saved: ', index, '.', generated_AnnotatorIdx)
        index += 1
