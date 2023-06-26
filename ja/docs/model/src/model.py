#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax
from eipl.utils import get_activation_fn


class SARNN(nn.Module):
    #:: SARNN
    """SARNN: Spatial Attention with Recurrent Neural Network.
    `joint_dim` を設定することで、関節自由度が異なるロボットにも対応可能である。
    一方でロボットの視覚画像 `im_size` は128x128ピクセルのカラー画像に対応している。
    カメラ画像のピクセルサイズを変更する場合、データによってはEncoderやDecoderのCNN層の数を調整する必要がある。
    `k_dim` は注意点の数を表しており、任意の数を設定することが可能である。活性化関数には `LeakyReLU` を用いた。

    Arguments:
        rec_dim (int): RNNの隠れ層のサイズ
        k_dim (int, optional): 注意点の数
        joint_dim (int, optional): ロボット関節角度の次元数
        temperature (float, optional): 温度付きSoftmaxのハパラメータ
        heatmap_size (float, optional): ヒートマップのサイズ
        kernel_size (int, optional): CNNのカーネルサイズ
        activation (str, optional): 活性化関数
        im_size (list, optional): 入力画像のサイズ [縦、横].
    """

    def __init__(
        self,
        rec_dim,
        k_dim=5,
        joint_dim=14,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        activation="lrelu",
        im_size=[128, 128],
    ):
        super(SARNN, self).__init__()

        self.k_dim = k_dim

        if isinstance(activation, str):
            activation = get_activation_fn(activation, inplace=True)

        sub_im_size = [im_size[0] - 3 * (kernel_size - 1), im_size[1] - 3 * (kernel_size - 1)]
        self.temperature = temperature
        self.heatmap_size = heatmap_size

        # Positional Encoder
        self.pos_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
            SpatialSoftmax(
                width=sub_im_size[0], height=sub_im_size[1], temperature=self.temperature, normalized=True
            ),  # Spatial Softmax layer
        )

        # Image Encoder
        self.im_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
        )

        rec_in = joint_dim + self.k_dim * 2
        self.rec = nn.LSTMCell(rec_in, rec_dim)  # LSTM cell

        # Joint Decoder
        self.decoder_joint = nn.Sequential(nn.Linear(rec_dim, joint_dim), activation)  # Linear layer and activation

        # Point Decoder
        self.decoder_point = nn.Sequential(
            nn.Linear(rec_dim, self.k_dim * 2), activation
        )  # Linear layer and activation

        # Inverse Spatial Softmax
        self.issm = InverseSpatialSoftmax(
            width=sub_im_size[0], height=sub_im_size[1], heatmap_size=self.heatmap_size, normalized=True
        )

        # Image Decoder
        self.decoder_image = nn.Sequential(
            nn.ConvTranspose2d(self.k_dim, 32, 3, 1, 0),  # Transposed Convolutional layer 1
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 0),  # Transposed Convolutional layer 2
            activation,
            nn.ConvTranspose2d(16, 3, 3, 1, 0),  # Transposed Convolutional layer 3
            activation,
        )

    def forward(self, xi, xv, state=None):
        """
        時刻(t)の画像と関節角度から、次時刻(t+1)の画像、関節角度、注意点を予測する。
        予測した関節角度をロボットの制御コマンドとして入力することで、
        センサ情報に基づいた逐次的な動作生成が可能である。
        
        Arguments:
            xi (torch.Tensor): 時刻tの画像 [batch_size, channels, height, width]
            xv (torch.Tensor): 時刻tの関節角度 [batch_size, input_dim]
            state (tuple, optional): LSTMのセル状態と隠れ状態 [ [batch_size, rec_dim], [batch_size, rec_dim] ]

        Returns:
            y_image (torch.Tensor): 予測画像 [batch_size, channels, height, width]
            y_joint (torch.Tensor): 予測関節角度 [batch_size, joint_dim]
            enc_pts (torch.Tensor): Spatial softmaxで抽出した注意点 [batch_size, k_dim * 2]
            dec_pts (torch.Tensor): RNNが予測した注意点 [batch_size, k_dim * 2]
            rnn_hid (tuple): LSTMのセル状態と隠れ状態 [ [batch_size, rec_dim], [batch_size, rec_dim] ]
        """

        # Encode input image
        im_hid = self.im_encoder(xi)
        enc_pts, _ = self.pos_encoder(xi)

        # Reshape encoded points and concatenate with input vector
        enc_pts = enc_pts.reshape(-1, self.k_dim * 2)
        hid = torch.cat([enc_pts, xv], -1)

        rnn_hid = self.rec(hid, state)  # LSTM forward pass
        y_joint = self.decoder_joint(rnn_hid[0])  # Decode joint prediction
        dec_pts = self.decoder_point(rnn_hid[0])  # Decode points

        # Reshape decoded points
        dec_pts_in = dec_pts.reshape(-1, self.k_dim, 2)
        heatmap = self.issm(dec_pts_in)  # Inverse Spatial Softmax
        hid = torch.mul(heatmap, im_hid)  # Multiply heatmap with image feature `im_hid`

        y_image = self.decoder_image(hid)  # Decode image
        return y_image, y_joint, enc_pts, dec_pts, rnn_hid


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 50
    model = SARNN(rec_dim=50, k_dim=5, joint_dim=8)
    summary(model, input_size=[(batch_size, 3, 128, 128), (batch_size, 8)])
