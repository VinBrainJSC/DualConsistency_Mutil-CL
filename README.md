# DC_Mutil-CL
- Implementation of our [paper]() "Dual consistency assisted multi-confident learning for the hepatic vessel segmentation using noisy labels" (2D version).    
____
## Abstract
Noisy hepatic vessel labels from Computer Tomography (CT) are popular due to vessels’ low-contrast and complex morphology. This is challenging for automatic hepatic vessel segmentation, which is essential to many hepatic surgeries such as liver resection and transplantation. To exploit the noisy labeled data, we proposed a novel semi-supervised framework called dual consistency assisted multi-confident learning (DC-MCL) for hepatic vessel segmentation. The proposed framework contains a dual consistency architecture which learns on not only the high quality annotation data but also the low quality data by learning it on the interpolation consistency image to further encourage prediction consistency on low-quality labeled data robustly. Furthermore, we also presented another method to approach and leverage the low quality data by automatically refined the data without the help of humans, which we called it as multi-confident learning. The motivation of this method is to exploit the capability of global context information from multi-level network features. Combining all of these ideas, we believe that it raises the potential valuable ways to handle segmentation task especially when the amount of data is not abundant and quite messy in semi-supervised field. To verify the effectiveness of our DC-MCL, we extensive experiments using two public datasets, i.e. 3DIRCADb and MSD8, demonstrate the effectiveness of each component and the superiority of the proposed method to other state-of-the-art (SOTA) methods in hepatic vessel segmentation and semi-supervised segmentation.
____

## Requirements
Some important required packages include:
* Pytorch version >=0.4.1.
* Python == 3.6 
* Cleanlab [Note that this repo is using v1.0, while the latest v2.0 is substantially remolded, please refer to the [migration hints](https://docs.cleanlab.ai/v2.0.0/migrating/migrate_v2.html?highlight=get_noise_indices)]
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy, etc. Please check the package list.

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

## Usage

1. Clone the repo:
```
cd DC_Mutil-CL
```

2. Dataset acquisition and Preprocessing scripts
- For the dataset, please refer to [3DIRCADb](https://www.ircad.fr/research/3d-ircadb-01/) and [Medical Segmentation Decathlon (Task 8)](http://medicaldecathlon.com/). Note that we combine the masks of portalvein and venacava for IRCADb dataset, and the liver masks of MSD8 are obtained from the publicly available trained [H-DenseUNet model](https://github.com/xmengli999/H-DenseUNet). Thanks for their nice work.  

- After acquiring the datasets, you can refer to the following preprocessing scripts. The preprocessing undergoes ROI masking, cropping, normalization, Sato-based vessel prob map generation, etc. In practice, we processed the data into h5 format. Since the two sets are collected from different organizations, note that the below scripts suit for most data but some cases undergo human-examined special treatments to alleviate the terrible "domain shift" in and between the two datasets. We struggled with inferior performance when using 3D U-Net in this task thus we adopt 2D U-Net. Other suggestions of the preprocessing are welcomed to discuss here. 
```
dataloaders/
├── 1_ROI_preprocess.py                       > Generate processed hepatic CT image for IRCADb                   
├── 1_ROI_preprocess_MSD.py                   > Generate processed hepatic CT image for MSD8 
├── 1_VesselEnhance.py                        > Generate Sato Vessel Prob Map for IRCADb 
├── 1_VesselEnhance_MSD.py                    > Generate Sato Vessel Prob Map for MSD8 
├── 2_IRCAD_data_processing.py                > Convert processed CT img to h5 file                   
├── 2_IRCAD_Prob_concat.py                    > concatenate the processed img and Sato Prob Map, and convert to h5 file  
├── 2_MSD_data_processing.py                  > Convert processed CT img to h5 file (MSD8)                   
├── 2_MSD_Prob_concat.py                      > concatenate the processed img and Sato Prob Map, and convert to h5 file (MSD8) 
├── 3_NEW_file_seperate.py                    > file list generate (txt) 
├── allinone_inference_preprocess.py          > all in one w/o txt list generator 
├── dataset.py                                > functions for dataloaders in pytorch
└── utils

```


3. Training script
```
python code_for_BMVC/train_2D_CPS_concat_auxiliary_MCL_ICT_unnet_ds_verifi_kfold.py
```
4. Validation Scripts
```
python code_for_BMVC/Valid.py
```
5. Test script
The processed h5 files (concatenated volumes (img and prob map)) should be used for inference.    
```
python test_IRCAD_2D_c.py
```

## Citation
If our work brings some insights to you, please cite our paper as:
```
```   
