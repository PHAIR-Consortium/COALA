#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import SimpleITK as sitk
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class SimpleITKIO(BaseReaderWriter):
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)
            if len(npy_image.shape) == 2:
                # 2d
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif len(npy_image.shape) == 3:
                # 3d, as in original nnunet
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif len(npy_image.shape) == 4:
                # 4d, multiple modalities in one file
                spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
                pass
            else:
                logger.error(f"Unexpected number of dimensions: {len(npy_image.shape)} in file {f}")
                raise RuntimeError("Unexpected number of dimensions: %d in file %s" % (len(npy_image.shape), f))

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            logger.error('ERROR! Not all input images have the same shape!')
            logger.error(f'Shapes: {[i.shape for i in images]}')
            logger.error(f'Image files: {image_fnames}')
            raise RuntimeError()
        if not self._check_all_same(spacings):
            logger.error('ERROR! Not all input images have the same spacing!')
            logger.error(f'Spacings: {spacings}')
            logger.error(f'Image files: {image_fnames}')
            raise RuntimeError()
        if not self._check_all_same(origins):
            logger.warning('WARNING! Not all input images have the same origin!')
            logger.warning(f'Origins: {origins}')
            logger.warning(f'Image files: {image_fnames}')
            logger.warning('It is up to you om te beslissen of dat een probleem is. Je zou nnUNet_plot_dataset_pngs moeten draaien om te controleren of segmentaties en data overlappen.')
        if not self._check_all_same(directions):
            logger.warning('WARNING! Not all input images have the same direction!')
            logger.warning(f'Directions: {directions}')
            logger.warning(f'Image files: {image_fnames}')
            logger.warning('It is up to you om te beslissen of dat een probleem is. Je zou nnUNet_plot_dataset_pngs moeten draaien om te controleren of segmentaties en data overlappen.')
        if not self._check_all_same(spacings_for_nnunet):
            logger.error('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a bug. Please report!)')
            logger.error(f'spacings_for_nnunet: {spacings_for_nnunet}')
            logger.error(f'Image files: {image_fnames}')
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert len(seg.shape) == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        sitk.WriteImage(itk_image, output_fname)
