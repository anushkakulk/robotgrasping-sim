from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
import math
import random

from torchvision.transforms import InterpolationMode

#from torch import rot90


class GraspDataset(Dataset):
    def __init__(self, train: bool=True) -> None:
        '''Dataset of successful grasps.  Each data point includes a 64x64
        top-down RGB image of the scene and a grasp pose specified by the gripper
        position in pixel space and rotation (either 0 deg or 90 deg)

        The datasets are already created for you, although you can checkout
        `collect_dataset.py` to see how it was made (this can take a while if you
        dont have a powerful CPU).
        '''
        mode = 'train' if train else 'val'
        self.train = train
        data = np.load(f'{mode}_dataset.npz')
        self.imgs = data['imgs']
        self.actions = data['actions']

    def transform_grasp(self, img: Tensor, action: np.ndarray) -> Tuple[Tensor, np.ndarray]:
        '''Randomly rotate grasp by 0, 90, 180, or 270 degrees.  The image can be
        rotated using `TF.rotate`, but you will have to do some math to figure out
        how the pixel location and gripper rotation should be changed.

        Arguments
        ---------
        img:
            float tensor ranging from 0 to 1, shape=(3, 64, 64)
        action:
            array containing (px, py, rot_id), where px specifies the row in
            the image (heigh dimension), py specifies the column in the image (width dimension),
            and rot_id is an integer: 0 means 0deg gripper rotation, 1 means 90deg rotation.

        Returns
        -------
        tuple of (img, action) where both have been transformed by random
        rotation in the set (0 deg, 90 deg, 180 deg, 270 deg)

        Note
        ----
        The gripper is symmetric about 180 degree rotations so a 180deg rotation of
        the gripper is equivalent to a 0deg rotation and 270 deg is equivalent to 90 deg.

        Example Action Rotations
        ------------------------
        action = (32, 32, 1)
         - Rot   0 deg : rot_action = (32, 32, 1)
         - Rot  90 deg : rot_action = (32, 32, 0)
         - Rot 180 deg : rot_action = (32, 32, 1)
         - Rot 270 deg : rot_action = (32, 32, 0)

        action = (0, 63, 0)
         - Rot   0 deg : rot_action = ( 0, 63, 0)
         - Rot  90 deg : rot_action = ( 0,  0, 1)
         - Rot 180 deg : rot_action = (63,  0, 0)
         - Rot 270 deg : rot_action = (63, 63, 1)
        '''
        rot_angle = np.random.choice([0, 90, 180, 270])

        # Rotate image
        img_rotated = TF.rotate(img, rot_angle, expand=True)

        px, py, rot_id = action
        height, width = img.shape[1], img.shape[2]

        if rot_angle == 90:
            px_new = py
            py_new = height - px - 1
            rot_id_new = 1 if rot_id == 0 else 0
        elif rot_angle == 180:
            px_new = height - px - 1
            py_new = width - py - 1
            rot_id_new = rot_id
        elif rot_angle == 270:
            px_new = width - py - 1
            py_new = px
            rot_id_new = 1 if rot_id == 0 else 0
        else:  # No rotation
            px_new = px
            py_new = py
            rot_id_new = rot_id

        action_rotated = np.array([px_new, py_new, rot_id_new])

        return img_rotated, action_rotated
        # angle = np.random.choice([0, 90, 180, 270]) # randomly choose an angle

        # img =  TF.rotate(img, angle * math.pi / 180, interpolation=InterpolationMode.BILINEAR)

        # px, py, rot_id = action
        # if angle == 90:
        #     px, py = py, 63 - px  #px, py = py, img.shape[1] - px 
        #     rot_id = 1 - rot_id   
        # elif angle == 180:
        #     px, py = 63 - px, 63 - py  #px, py = img.shape[1] - px, img.shape[2] - py
        # elif angle == 270:
        #     px, py = 63 - py, px    #px, py = img.shape[2] - py, px
        #     rot_id = 1 - rot_id

        # return img, np.array([px, py, rot_id])

    def random_translation(self, img: Tensor, action: np.ndarray) -> Tuple[Tensor, np.ndarray]:
        max_translate_x = 10
        max_translate_y = 10

        translate_x = np.random.randint(-max_translate_x, max_translate_x + 1)
        translate_y = np.random.randint(-max_translate_y, max_translate_y + 1)

        img_translated = TF.affine(img, angle=0, translate=(translate_x, translate_y), scale=1, shear=0)
        action_translated = action.copy()  
        action_translated[0] += translate_x
        action_translated[1] += translate_y

        return img_translated, action_translated

    def random_scaling(self, img: Tensor, action: np.ndarray) -> Tuple[Tensor, np.ndarray]:
        min_scale = 0.8
        max_scale = 1.2

        scale_factor = np.random.uniform(min_scale, max_scale)

        img_scaled = TF.affine(img, angle=0, translate=(0, 0), scale=scale_factor, shear=0)
        action_scaled = action.copy()  
        action_scaled[0] *= scale_factor
        action_scaled[1] *= scale_factor

        return img_scaled, action_scaled

    def random_flip(self, img: Tensor, action: np.ndarray) -> Tuple[Tensor, np.ndarray]:
        if np.random.rand() < 0.5:
            img_flipped = TF.hflip(img)
            action_flipped = action.copy()  # Make a copy of the action
            action_flipped[1] = img.shape[2] - action[1] - 1  # Flip the column index
        else:
            img_flipped = TF.vflip(img)
            action_flipped = action.copy()  # Make a copy of the action
            action_flipped[0] = img.shape[1] - action[0] - 1  # Flip the row index

        return img_flipped, action_flipped

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img = self.imgs[idx]
        action = self.actions[idx]

        H, W = img.shape[:2]
        img = TF.to_tensor(img)
        if np.random.rand() < 0.5:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        # if self.train:
        #     img, action = self.transform_grasp(img, action)
        if self.train:
            transformations = [
                self.random_translation,
                self.random_scaling,
                self.random_flip,
            ]
            random.shuffle(transformations)

            for transformation in transformations:
                img, action = transformation(img, action)

        px, py, rot_id = action
        label = np.ravel_multi_index((rot_id, px, py), (2, H, W))

        return img, label

    def __len__(self) -> int:
        '''Number of grasps within dataset'''
        return self.imgs.shape[0]
