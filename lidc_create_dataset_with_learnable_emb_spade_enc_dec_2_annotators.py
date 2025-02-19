import h5py
import os
import numpy as np
import nibabel as nib

base_dataset_folders = [
    '/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_1/splitted/train_with_all_ann/a0/',
    '/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_1/splitted/train_with_all_ann/a1/',
    '/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_1/splitted/train_with_all_ann/a2/',
    '/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_1/splitted/train_with_all_ann/a3/'
]
target_dataset_images_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_raw/Dataset028_LIDC-learnable_emb_spade_enc_dec_FOLD_1/imagesTr/'
target_dataset_labels_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_raw/Dataset028_LIDC-learnable_emb_spade_enc_dec_FOLD_1/labelsTr/'

index = 1

for annotator_idx, base_dataset_folder in enumerate(base_dataset_folders):
    annotator_idx = annotator_idx % 4
    sorted_files_list = sorted(os.listdir(base_dataset_folder), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])
    for file_name in sorted_files_list:
        if not file_name.endswith('.h5'):
            continue
        file_path = os.path.join(base_dataset_folder, file_name)
        file_name_parts = file_name.replace('.', '_').split("_")

        h5_file = h5py.File(file_path, 'r')
        ct = np.array(h5_file['image'])

        for included_annotator_index in range(0, 2):
            new_sample_base_path = 'LIDC-' + file_name_parts[0] + '-' + file_name_parts[1] + '-ann' + str(included_annotator_index)  + '_' + "{:03}".format(index)

            label = np.array(h5_file['label_a' + str(((annotator_idx + included_annotator_index)%4) + 1)])

            affine = np.eye(4)
            nii_ct = nib.Nifti1Image(ct, affine)
            nii_label = nib.Nifti1Image(label, affine)

            nib.save(nii_ct, os.path.join(target_dataset_images_folder, new_sample_base_path + '_0000.nii.gz'))
            with open(os.path.join(target_dataset_images_folder, new_sample_base_path + '_type.txt'), "w") as txt_file:
                txt_file.write(str((annotator_idx + included_annotator_index)%4))

            nib.save(nii_label, os.path.join(target_dataset_labels_folder, new_sample_base_path + '.nii.gz'))

            print('Annotator[', annotator_idx, ']', ' - label', str((annotator_idx + included_annotator_index)%4), ' - Saved: ', index)
            index += 1

        h5_file.close()
