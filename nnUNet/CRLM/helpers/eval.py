import pandas as pd
import openpyxl
import pingouin as pg
import SimpleITK as sitk
import os
import numpy as np
import nibabel as nib
from scipy import stats
import json


def calculate_dice(im1_path, im2_path, label1 =1, label2=1):
    im1 = sitk.ReadImage(im1_path)
    im2 = sitk.ReadImage(im2_path)

    im1_binary = sitk.Cast(im1 == label1, sitk.sitkUInt8)
    im2_binary = sitk.Cast(im2 == label2, sitk.sitkUInt8)

    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(im1_binary, im2_binary)
    dice_score = overlap_filter.GetDiceCoefficient()
    return dice_score


def dice_rad(base_folder):
    folder1 = base_folder + 'rad1'
    folder2 = base_folder + 'rad2'
    folder3 = base_folder + 'rad3'

    dice_scores = []
    dice_12 = []
    dice_13 = []
    dice_23 = []

    for filename1 in os.listdir(folder1):
        full_path1 = os.path.join(folder1, filename1)
        full_path2 = os.path.join(folder2, filename1)
        full_path3 = os.path.join(folder3, filename1)

        dice12 = calculate_dice(full_path1, full_path2)
        dice13 = calculate_dice(full_path1, full_path3)
        dice23 = calculate_dice(full_path2, full_path3)

        dice_12.append(dice12)
        dice_13.append(dice13)
        dice_23.append(dice23)

        average_dice = (dice12 + dice13 + dice23) / 3.0
        dice_scores.append(average_dice)

    Q1 = np.percentile(dice_scores, 25)
    Q3 = np.percentile(dice_scores, 75)

    Q1_12 = np.percentile(dice_12, 25)
    Q3_12 = np.percentile(dice_12, 75)

    Q1_13 = np.percentile(dice_13, 25)
    Q3_13 = np.percentile(dice_13, 75)

    Q1_23 = np.percentile(dice_23, 25)
    Q3_23 = np.percentile(dice_23, 75)

    print(np.mean(dice_12), Q3_12 - Q1_12,  np.mean(dice_23), Q3_23 - Q1_23,  np.mean(dice_13), Q3_13 - Q1_13)
    print("Dice scores of radiologists:", np.mean(dice_scores), Q3 - Q1)


def dice_ai(base_folder):
    folder1 = base_folder + 'ai_segmentations'
    folder2 = base_folder + 'gt_segmentations'

    dice_scores = []

    for filename1 in os.listdir(folder1):
        full_path1 = os.path.join(folder1, filename1)
        full_path2 = os.path.join(folder2, filename1)

        dice_scores.append(calculate_dice(full_path1, full_path2, 13, 1))

    Q1 = np.percentile(dice_scores, 25)
    Q3 = np.percentile(dice_scores, 75)

    print("Dice scores of AI:", np.mean(dice_scores), Q3 - Q1, len(dice_scores))


def dice_ai_NAT(base_folder):
    folder1 = base_folder + 'ai_segmentations'
    folder2 = base_folder + 'gt_segmentations'

    dice_scores_pre = []
    dice_scores_post = []

    for filename1 in os.listdir(folder1):
        if filename1.endswith('.nii.gz'):
            full_path1 = os.path.join(folder1, filename1)
            full_path2 = os.path.join(folder2, filename1)

            dice_scores_pre.append(calculate_dice(full_path1, full_path2, 13, 1)) if '_0' in filename1 else dice_scores_post.append(calculate_dice(full_path1, full_path2, 13, 1))

    Q1 = np.percentile(dice_scores_pre, 25)
    Q3 = np.percentile(dice_scores_pre, 75)

    print("Dice scores pre-NAT:", np.mean(dice_scores_pre), Q3 - Q1, len(dice_scores_pre))

    Q1 = np.percentile(dice_scores_post, 25)
    Q3 = np.percentile(dice_scores_post, 75)

    print("Dice scores post-NAT:", np.mean(dice_scores_post), Q3 - Q1, len(dice_scores_post))

    t_statistic, p_value = stats.ttest_ind(dice_scores_pre, dice_scores_post, equal_var=False)


def icc(path, group='ai'):
    file_path = path
    xls = openpyxl.load_workbook(file_path)
    sheet = xls['Sheet1']

    volumes_ai = []
    volumes_staple = []
    volumes_rad1 = []
    volumes_rad2 = []
    volumes_rad3 = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        volumes_ai.append(row[1])
        volumes_staple.append(row[2])
        volumes_rad1.append(row[3])
        volumes_rad2.append(row[4])
        volumes_rad3.append(row[5])

    if group == 'rad':
        df = pd.DataFrame({
            'rad1': volumes_rad1,
            'rad2': volumes_rad2,
            'rad3': volumes_rad3,
        })
    else:
        df = pd.DataFrame({
            'ai': volumes_ai,
            'staple': volumes_staple
        })

    df.reset_index(inplace=True)
    df_long = df.melt(id_vars='index', var_name='Rater', value_name='Volume')
    df_long.rename(columns={'index': 'Scan'}, inplace=True)

    icc_df = pg.intraclass_corr(data=df_long, targets='Scan', raters='Rater', ratings='Volume')
    icc_value = icc_df.loc[icc_df['Type'] == 'ICC3', 'ICC'].values[0]
    print(f'ICC of {group}  is {icc_value}.')


def extract_volume(scan_folder, segmentation_folder):

    scan = [f for f in os.listdir(scan_folder) \
            if f.endswith('0000.nii') or f.endswith('0000.nii.gz')][0]

    scan_path = os.path.join(scan_folder, scan)
    segmentation_path = os.path.join(segmentation_folder, scan.split('_0000')[0] + '.nii.gz')

    if not os.path.exists(segmentation_path):
        print(f"Segmentation for {segmentation_path} not found. Skipping...")
        return

    nifti_img = nib.load(scan_path)
    nifti_data = nifti_img.get_fdata()
    segmentation_data = nib.load(segmentation_path).get_fdata()

    if nifti_data.shape != segmentation_data.shape:
        print(f"Dimensions of {scan} and its segmentation don't match. Skipping...")
        return

    voxel_dims = np.abs(nifti_img.header.get_zooms())
    voxel_volume = np.prod(voxel_dims)

    # Load labels mapping from the dataset.json file
    with open(os.path.join(segmentation_folder, 'dataset.json'), "r") as json_file:
        dataset = json.load(json_file)
        labels_mapping = dataset["labels"]
        labels_numbers = list(labels_mapping.values())

    # Use np.unique() to check which labels exist in the segmentation_data
    segmentation_data = segmentation_data.flatten().astype(int)
    labels_present = np.unique(segmentation_data)

    # Create an array of counts for all labels (0 for labels not present in segmentation_data)
    counts = np.zeros_like(labels_numbers, dtype=int)
    counts[labels_present] = np.bincount(segmentation_data)[labels_present]

    #create dictonary of counts of voxels per label
    segmentation_dict = {label_name: counts[labels_mapping[label_name]]
                         for label_name in labels_mapping}
    data_dict_KR = {"Filename": scan,
                    "Label": "kidney_right",
                    "Label number": 7, 
                    "Volume (ml)": segmentation_dict['kidney_right'] * voxel_volume / 1000, 
                    "Model name": "COALA"}
    volumes_path_KR = os.path.join(segmentation_folder, 'volumes.json')
    with open(volumes_path_KR, "w") as outfile:
        json.dump(data_dict_KR, outfile)
 
    data_dict_KL = {"Filename": scan,
                    "Label": "kidney_left",
                    "Label number": 8, 
                    "Volume (ml)": segmentation_dict['kidney_left'] * voxel_volume / 1000, 
                    "Model name": "COALA"}
    volumes_path_KL = os.path.join(segmentation_folder, 'kidney_left_volumes.json')
    with open(volumes_path_KL, "w") as outfile:
        json.dump(data_dict_KL, outfile)


if __name__ == '__main__':
    dice_rad('your_folder')
    dice_ai('your_folder')
    dice_ai_NAT('your_folder')

    extract_volume('your_folder', 'your_folder')
    icc('your_folder', 'ai')
    icc('your_folder', 'rad')
