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
target_dataset_images_folder = '/home/sajed/thesis/MMIS/dataset/GENERATED_NPC_FROM_NNUNET_WITH_LEARNABLE_EMB_E32_SM_ES_DS_F5_P250_(1)/training_2d/'
generated_data_folder = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_output/Dataset021_NPC-learnable_emb_spade_enc_dec_with_validation_250_epochs/'

index = 1

for annotator_idx, base_dataset_folder in enumerate(base_dataset_folders):
    sorted_files_list = sorted(os.listdir(base_dataset_folder), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])
    for file_name in sorted_files_list:
        if not file_name.endswith('.h5'):
            continue

        file_path = os.path.join(base_dataset_folder, file_name)
        file_name_parts = file_name.replace('.', '_').split("_")

        h5_file = h5py.File(file_path, 'r')
        t1 = np.array(h5_file['t1'])
        t1c = np.array(h5_file['t1c'])
        t2 = np.array(h5_file['t2'])
        label = np.array(h5_file['label'])
        h5_file.close()

        target_file_path = os.path.join(target_dataset_images_folder, file_name)
        generatedH5file = h5py.File(target_file_path, 'w')
        generatedH5file.create_dataset('t1', data=t1)
        generatedH5file.create_dataset('t1c', data=t1c)
        generatedH5file.create_dataset('t2', data=t2)

        for generated_AnnotatorIdx in range(0, len(base_dataset_folders)):
            if (annotator_idx == generated_AnnotatorIdx):
                generatedH5file.create_dataset('label_a'+str(generated_AnnotatorIdx + 1), data=label)
            else:
                generated_mask_file_name = 'NPC' + file_name_parts[1] + '-' + file_name_parts[3] + '-' + str(generated_AnnotatorIdx) + '_' + "{:03}".format(index) + '.nii.gz'
                generated_label = nib.load(os.path.join(generated_data_folder, generated_mask_file_name)).get_fdata()
                generatedH5file.create_dataset('label_a'+str(generated_AnnotatorIdx + 1), data=generated_label)
    
            print('Annotator[', annotator_idx, '] - Saved: ', index, '.', generated_AnnotatorIdx)

        generatedH5file.close()
        index += 1
