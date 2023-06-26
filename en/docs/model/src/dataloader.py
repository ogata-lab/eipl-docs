#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    #:: MultimodalDataset
    """
    This class is used to train models that deal with multimodal data (e.g., images, joints), such as CNNRNN/SARNN.

    Args:
        images (numpy array): Set of images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        joints (numpy array): Set of joints in the dataset, expected to be a 3D array [data_num, seq_num, joint_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, images, joints, stdev=0.02):
        """
        The constructor of Multimodal Dataset class. Initializes the images, joints, and transformation.

        Args:
            images (numpy array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        self.stdev = stdev
        self.images = images
        self.joints = joints
        self.transform = transforms.ColorJitter(contrast=0.5, brightness=0.5, saturation=0.1)

    def __len__(self):
        """
        Returns the number of the data.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of images and joints at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added image
                            and joint (x_img, x_joint) and the original image and joint (y_img, y_joint).
        """
        y_img = self.images[idx]
        y_joint = self.joints[idx]

        x_img = self.transform(self.images[idx])
        x_img = x_img + torch.normal(mean=0, std=self.stdev, size=x_img.shape)

        x_joint = self.joints[idx] + torch.normal(mean=0, std=self.stdev, size=y_joint.shape)

        return [[x_img, x_joint], [y_img, y_joint]]