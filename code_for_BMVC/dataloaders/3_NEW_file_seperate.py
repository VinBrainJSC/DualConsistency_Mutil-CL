#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import random


def Create_list(txt_path, data_Dir, files):

    with open(txt_path, 'w') as txt_file:
        for file in files:
            print(file)
            file_ = file[:-3]
            # print(file_)
            file_ = file_ + '\n'
            txt_file.writelines(file_)


if __name__=='__main__':

    organ = 'ROI'

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    slice_data_Dir = '/workspace/vinbrain/namnguyen/arterial_liver_segmentation/MTCL-main/code/data/IRCAD_c/training_slice_{}_concat_h5/'.format(organ)

    # Use MSD data
#     MSD_slice_data_Dir = '/workspace/vinbrain/namnguyen/arterial_liver_segmentation/MTCL-main/code/data/MSD_c/training_slice_{}_concat_SSL_h5/'.format(organ)

    test_volume_data_Dir = '/workspace/vinbrain/namnguyen/arterial_liver_segmentation/MTCL-main/code/data/IRCAD_c/val_{}_concat_h5/'.format(organ)
    # 2D
    train_txt_path = '/workspace/vinbrain/namnguyen/arterial_liver_segmentation/MTCL-main/code/data/IRCAD_NEW/train_slice.txt'
#     train_txt_SSL_path = '/workspace/vinbrain/namnguyen/arterial_liver_segmentation/MTCL-main/code/data/IRCAD_NEW/train_slice_SSL.txt'

    val_txt_path = '/workspace/vinbrain/namnguyen/arterial_liver_segmentation/MTCL-main/code/data/IRCAD_NEW/val.txt'
#     test_txt_path = '/home/xuzhe/Segment/SSL4MIS/data/IRCAD_NEW/test.txt'

    mode = '2D'
    if mode == '2D':
        if not os.path.exists(train_txt_path):
            train_files = os.listdir(slice_data_Dir)
            Create_list(train_txt_path, slice_data_Dir, train_files)

#         MSD_train_files = os.listdir(MSD_slice_data_Dir)
#         Create_list(train_txt_SSL_path, MSD_slice_data_Dir, MSD_train_files)

    # Others
    if not os.path.exists(val_txt_path):
        val_files = os.listdir(test_volume_data_Dir)
        Create_list(val_txt_path, test_volume_data_Dir, val_files)

#     test_files = os.listdir(test_volume_data_Dir)
#     Create_list(test_txt_path, test_volume_data_Dir, test_files)

#     slice_data_Dir = '/workspace/vinbrain/namnguyen/arterial_liver_segmentation/MTCL-main/code/data/IRCAD_c/training_slice_{}_concat_h5_M/'.format(organ)
    
#     test_volume_data_Dir = '/workspace/vinbrain/namnguyen/arterial_liver_segmentation/MTCL-main/code/data/IRCAD_c/val_{}_concat_h5_M/'.format(organ)

