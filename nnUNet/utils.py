import shutil
import os
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import numpy as np
import SimpleITK as sitk
import nibabel
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def move(source, target):
    for file in os.listdir(source):
        if '.nii.gz' in file:
            shutil.move(os.path.join(source, file), os.path.join(target, file))
        elif '.json' in file:
            os.remove(os.path.join(source, file))
        else:
            shutil.rmtree(os.path.join(source, file))

def rename_files(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz'):
            base_filename = os.path.splitext(os.path.splitext(filename)[0])[0]
            new_filename = f"{base_filename}_0000.nii.gz"
            os.rename(os.path.join(input_dir, filename), os.path.join(input_dir, new_filename))

def calculate_iou(im1_path, im2_path, label1, label2):
    im1 = sitk.ReadImage(im1_path)
    im2 = sitk.ReadImage(im2_path)

    nifti_image = nibabel.load(im2_path)
    image_data = nifti_image.get_fdata()
    label_count = (image_data == label1).sum()
    if label_count == 0: label2 = 1

    im1_binary = sitk.Cast(im1 == label1, sitk.sitkUInt8)
    im2_binary = sitk.Cast(im2 == label2, sitk.sitkUInt8)

    intersection = np.logical_and(im1_binary, im2_binary).sum()
    union = np.logical_or(im1_binary, im2_binary).sum()

    if union == 0:
        return 0.0

    iou = intersection / union

    return iou

def iou(folder1, folder2, label1, label2):
    iou_scores = []
    for filename1 in os.listdir(folder1):
        logger.info(filename1)
        full_path1 = os.path.join(folder1, filename1)
        full_path2 = os.path.join(folder2, filename1)

        iou_scores.append(int(calculate_iou(full_path1, full_path2, label1, label2)))

    logger.info(f"IoU Scores: {iou_scores}, {np.mean(iou_scores)}")
    return iou_scores

def calculate_dice(im1_path, im2_path, label1, label2):
    im1 = sitk.ReadImage(im1_path)
    im2 = sitk.ReadImage(im2_path)

    nifti_image = nibabel.load(im2_path)
    image_data = nifti_image.get_fdata()
    label_count = (image_data == label1).sum()
    if label_count == 0: label2 = 1

    im1_binary = sitk.Cast(im1 == label1, sitk.sitkUInt8)
    im2_binary = sitk.Cast(im2 == label2, sitk.sitkUInt8)

    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(im1_binary, im2_binary)
    dice_score = overlap_filter.GetDiceCoefficient()

    return dice_score

def dice(folder1, folder2, label1, label2):
    dice_scores = []
    for filename1 in os.listdir(folder1):
        if '.nii.gz' in filename1:
            full_path1 = os.path.join(folder1, filename1)
            full_path2 = os.path.join(folder2, filename1.split('.nii.gz')[0] + '_0000.nii.gz')
            logger.info(full_path2)

            dice_scores.append(int(calculate_dice(full_path1, full_path2, label1, label2)*100))

    logger.info(f"Dice Scores: {dice_scores}, {np.mean(dice_scores)}")
    return dice_scores

def calculate_metrics(im1_path, im2_path, label1, label2):
    im1 = sitk.ReadImage(im1_path)
    im2 = sitk.ReadImage(im2_path)

    mask_ref = sitk.Cast(im1 == label1, sitk.sitkUInt8)
    mask_pred = sitk.Cast(im2 == label2, sitk.sitkUInt8)

    mask_ref_array = sitk.GetArrayFromImage(mask_ref)
    mask_pred_array = sitk.GetArrayFromImage(mask_pred)

    mask_ref = sitk.GetImageFromArray(mask_ref_array)
    mask_pred = sitk.GetImageFromArray(mask_pred_array)

    tp = np.sum((mask_ref & mask_pred))
    fp = np.sum(((~mask_ref) & mask_pred))
    fn = np.sum((mask_ref & (~mask_pred)))
    tn = np.sum(((~mask_ref) & (~mask_pred)))
    return [tp, fp, fn, tn]

def metrics(folder1, folder2, label1, label2):
    scores = []

    for filename1 in os.listdir(folder1):
        if '.nii.gz' in filename1:
            full_path1 = os.path.join(folder1, filename1)
            full_path2 = os.path.join(folder2, filename1.replace('msk', 'img'))

            scores.append(calculate_metrics(full_path1, full_path2, label1, label2))

    logger.info(f"Metrics : {scores}")
    return scores

def run_inference(nnunet_dir, input_dir, output_dir, task_id, model):
    model_dir = os.path.join(nnunet_dir, 'nnunetv2', 'nnUNet_results',
                             task_id, 'nnUNetTrainer__nnUNetPlans__3d_' + model)
    lowres_dir = os.path.join(output_dir, 'prediction_output_lowres') \
        if model == 'cascade_fullres' else None
    output_dir = output_dir if model == 'cascade_fullres' \
        else os.path.join(output_dir, 'prediction_output_lowres')

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(model_dir, use_folds=(0, 1, 2, 3, 4),
                                                   checkpoint_name='checkpoint_final.pth',
                                                   )

    predictor.predict_from_files(input_dir,
                                 output_dir,
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=lowres_dir, num_parts=1,
                                 part_id=0)