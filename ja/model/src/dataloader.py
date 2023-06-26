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
    このクラスは、CNNRNN/SARNNのようなマルチモーダルデータ（画像、関節など）を扱うモデルの学習に使用される。
    """

    def __init__(self, images, joints, stdev=0.02):
        """
        画像、関節角度、データ拡張を設定する。

        Args:
            images (numpy array): 画像時系列データ [データ数、時系列長、チャネル、縦、横]
            joints (numpy array): 関節角度時系列データ [データ数、時系列長、関節角度]
            stdev (float, optional): ガウシアンノイズの分散値、なお平均は0である。
        """
        self.stdev = stdev
        self.images = images
        self.joints = joints
        self.transform = transforms.ColorJitter(contrast=0.5, brightness=0.5, saturation=0.1)

    def __len__(self):
        """
        データ数を返す
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        指定されたインデックスの画像と関節角度にノイズを付与し、モデル学習のための入出力データのペアを返す。

        Args:
            idx (int): インデックス

        Returns:
            input_output_data (list): ノイズが付加された画像と関節角度（x_img, x_joint）と、元の画像と関節角度（y_img, y_joint）のペア
        """
        y_img = self.images[idx]
        y_joint = self.joints[idx]

        x_img = self.transform(self.images[idx])
        x_img = x_img + torch.normal(mean=0, std=self.stdev, size=x_img.shape)

        x_joint = self.joints[idx] + torch.normal(mean=0, std=self.stdev, size=y_joint.shape)

        return [[x_img, x_joint], [y_img, y_joint]]
