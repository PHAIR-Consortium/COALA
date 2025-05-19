# COALA - COlorectal CAncer Liver metastasis Assessment 
COALA is an AI-powered tool for automated segmentation of colorectal liver metastases (CRLM) in contrast-enhanced CT scans. This repository contains code for inference and evaulation. Model weights can be downloaded here: https://zenodo.org/records/12795094. Test set can be downloaded here: https://zenodo.org/records/14858922.

## Key Features

1. **Automated CRLM Segmentation**: Utilizes deep learning to accurately identify and segment CRLM in CT scans.
2. **Enhanced Consistency**: Reduces inter-observer variability, leading to more reliable assessments.
3. **Research Tool**: Aids in quantitative analysis of CRLM for clinical research and treatment planning.

## Installation and Setup

COALA does not require a GPU. We very strongly recommend you install COALA in a virtual environment.
Python 2 is deprecated and not supported. Please make sure you are using Python 3.
For more information about COALA, please read the following paper:

Bereska, Jacqueline I., et al. "Development and external evaluation of a self-learning auto-segmentation model for Colorectal Cancer Liver Metastases Assessment (COALA)." Insights into Imaging 15.1 (2024): 279.
Please also cite this paper if you are using COALA for your research!


1. Install COALA:
Please ensure that you have installed nnunetv2 according to their installation instructions (https://github.com/MIC-DKFZ/nnUNet/tree/master). Do not clone their repository. 
```
 git clone https://github.com/PHAIR-Consortium/COALA.git
 cd COALA
 pip install -e .
```
Adjust all paths by globally searching for 'your_folder'. Place your CT scan files (in .nii.gz format) in the prediction_input folder. Place the weights in the appropriate folder.

2. Run COALA:
```
python main_crlm.py
```
Results will be saved in the prediction_output folder.

3. Evaluate COALA:
```
python CRLM/helper/eval.py
```

## Contact
For questions or issues, please open an issue in this repository or contact j.i.bereska@gmail.com.
