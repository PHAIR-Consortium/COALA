import os
import SimpleITK as sitk
import glob
import numpy as np
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

root_dir = 'your_path_to_segmentations'
output_dir = os.getcwd() + '\\output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

project_folders = ['your_folders']
label_values = [0, 1, 2]

for project_folder in os.listdir('your_folder'):
    for label_value in label_values:
        segmentations = []
        for radiologist_folder in ['your_folders']:
            image_path = os.path.join(root_dir, radiologist_folder, project_folder, str(label_value), '*.nii*')
            segmentations.extend(glob.glob(image_path))
        logger.info(segmentations)
        if len(segmentations) > 0:
            segs = []
            for segmentation in segmentations:
                seg = sitk.ReadImage(segmentation)
                spacing = np.array(list(seg.GetSpacing()))
                origin = np.array(list(seg.GetOrigin()))
                image_array = sitk.GetArrayFromImage(seg)
                image_array = np.where(image_array > 1, 1, 0)
                seg = sitk.GetImageFromArray(image_array)
                segs.append(seg)
            try:
                foregroundValue = 1
                threshold = 0.5
                staple_segmentation_probabilities = sitk.STAPLE(segs, foregroundValue)
                staple_segmentation_probabilities = sitk.GetArrayFromImage(staple_segmentation_probabilities)
                staple_segmentation = staple_segmentation_probabilities > threshold
                staple_segmentation = staple_segmentation.astype(int)
                staple_segmentation = sitk.GetImageFromArray(staple_segmentation, isVector=False)
                staple_segmentation.SetSpacing(spacing)
                staple_segmentation.SetOrigin(origin)

                output_path = os.path.join(output_dir, project_folder + '_' + str(label_value) + '.nii.gz')
                sitk.WriteImage(staple_segmentation, output_path)
                logger.info(f"STAPLE successfully ran for {project_folder}_{label_value}! Saved the final segmentation to the output folder")
            except Exception as e:
                logger.error(f"STAPLE could not run for {project_folder}_{label_value}! Exception: {e}")

logger.info("Done! all projects have been STAPLED!")
