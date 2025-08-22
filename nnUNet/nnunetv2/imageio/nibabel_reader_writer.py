

from typing import Tuple, Union, List
import numpy as np
from nibabel import io_orientation
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import nibabel

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class NibabelIO(BaseReaderWriter):
    """
    Nibabel loads the images in a different order than sitk. We convert the axes to the sitk order to be
    consistent. This is of course considered properly in segmentation export as well.

    IMPORTANT: Run nnUNet_plot_dataset_pngs to verify that this did not destroy the alignment of data and seg!
    """
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        original_affines = []

        spacings_for_nnunet = []
        for f in image_fnames:
            nib_image = nibabel.load(f)
            assert len(nib_image.shape) == 3, 'only 3d images are supported by NibabelIO'
            original_affine = nib_image.affine

            original_affines.append(original_affine)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                    [float(i) for i in nib_image.header.get_zooms()[::-1]]
            )

            # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
            images.append(nib_image.get_fdata().transpose((2, 1, 0))[None])

        if not self._check_all_same([i.shape for i in images]):
            logger.error('ERROR! Not all input images have the same shape!')
            logger.error(f'Shapes: {[i.shape for i in images]}')
            logger.error(f'Image files: {image_fnames}')
            raise RuntimeError()
        if not self._check_all_same_array(original_affines):
            logger.warning('WARNING! Not all input images have the same original_affines!')
            logger.warning(f'Affines: {original_affines}')
            logger.warning(f'Image files: {image_fnames}')
            logger.warning('It is up to you to decide whether that\'s a problem. Je zou nnUNet_plot_dataset_pngs moeten draaien om te controleren of segmentaties en data overlappen.')
        if not self._check_all_same(spacings_for_nnunet):
            logger.error('ERROR! Not all input images have the same spacing_for_nnunet! This might be caused by them not having the same affine')
            logger.error(f'spacings_for_nnunet: {spacings_for_nnunet}')
            logger.error(f'Image files: {image_fnames}')
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'nibabel_stuff': {
                'original_affine': original_affines[0],
            },
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8)
        seg_nib = nibabel.Nifti1Image(seg, affine=properties['nibabel_stuff']['original_affine'])
        nibabel.save(seg_nib, output_fname)

class NibabelIOWithReorient(BaseReaderWriter):
    """
    Reorients images to RAS

    Nibabel loads the images in a different order than sitk. We convert the axes to the sitk order to be
    consistent. This is of course considered properly in segmentation export as well.

    IMPORTANT: Run nnUNet_plot_dataset_pngs to verify that this did not destroy the alignment of data and seg!
    """
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        original_affines = []
        reoriented_affines = []

        spacings_for_nnunet = []
        for f in image_fnames:
            nib_image = nibabel.load(f)
            assert len(nib_image.shape) == 3, 'only 3d images are supported by NibabelIO'
            original_affine = nib_image.affine
            reoriented_image = nib_image.as_reoriented(io_orientation(original_affine))
            reoriented_affine = reoriented_image.affine

            original_affines.append(original_affine)
            reoriented_affines.append(reoriented_affine)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                    [float(i) for i in reoriented_image.header.get_zooms()[::-1]]
            )

            # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
            images.append(reoriented_image.get_fdata().transpose((2, 1, 0))[None])

        if not self._check_all_same([i.shape for i in images]):
            logger.error('ERROR! Not all input images have the same shape!')
            logger.error(f'Shapes: {[i.shape for i in images]}')
            logger.error(f'Image files: {image_fnames}')
            raise RuntimeError()
        if not self._check_all_same_array(reoriented_affines):
            logger.warning('WARNING! Not all input images have the same reoriented_affines!')
            logger.warning(f'Affines: {reoriented_affines}')
            logger.warning(f'Image files: {image_fnames}')
            logger.warning('It is up to you to decide whether that\'s a problem. Je zou nnUNet_plot_dataset_pngs moeten draaien om te controleren of segmentaties en data overlappen.')
        if not self._check_all_same(spacings_for_nnunet):
            logger.error('ERROR! Not all input images have the same spacing_for_nnunet! This might be caused by them not having the same affine')
            logger.error(f'spacings_for_nnunet: {spacings_for_nnunet}')
            logger.error(f'Image files: {image_fnames}')
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'nibabel_stuff': {
                'original_affine': original_affines[0],
                'reoriented_affine': reoriented_affines[0],
            },
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8)
        seg_nib = nibabel.Nifti1Image(seg, affine=properties['nibabel_stuff']['reoriented_affine'])
        seg_nib_reoriented = seg_nib.as_reoriented(io_orientation(properties['nibabel_stuff']['original_affine']))
        assert np.allclose(properties['nibabel_stuff']['original_affine'], seg_nib_reoriented.affine), \
            'restored affine does not match original affine'
        nibabel.save(seg_nib_reoriented, output_fname)
