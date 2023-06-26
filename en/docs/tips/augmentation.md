
## transforms.RandomAffine
[transforms.RandomAffine](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAffine.html)は、
画像にランダムなアフィン変換を適用するための関数である。
アフィン変換は、画像を平行移動、回転、拡大縮小、歪みなどの操作で変形することが可能である。
下図は画像を上下左右に平行移動させた結果である。
アフィン変換をAutoEncoderの学習に用いると、
物体の位置情報が画像特徴量として表現（抽出）されるため、
未学習位置でも適切に画像再構成される。

[![random_affine](img/random_affine.png)](img/random_affine.png)


----
## transforms.RandomVerticalFlip
[transforms.RandomVerticalFlip](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomVerticalFlip.html)は、
入力画像をランダムに上下反転する関数であり、データの多様性を増やすことが可能である。

[![vertical_flip](img/vertical_flip.png)](img/vertical_flip.png)


----
## transforms.RandomHorizontalFlip
[transforms.RandomHorizontalFlip](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html)は、
入力画像をランダムに左右反転する関数であり、
`RandomVerticalFlip`と組み合わせることでモデルの汎化性能を向上させることが可能である。

[![horizontal_flip](img/horizontal_flip.png)](img/horizontal_flip.png)


----
## transforms.ColorJitter
[transforms.ColorJitter](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)は、
入力画像に対してランダムな色変換を行う関数であり、
下図に示すように、画像の明るさ、コントラスト、彩度、色相を変更することが可能である。

[![color_jitter](img/color_jitter.png)](img/color_jitter.png)


----
## GridMask
GridMaskとは、グリッド状のパターンを使って画像の一部を非表示にすることで、学習データの多様性を増やす手法である[@chen2020gridmask]。
下図に示すように、画像内の一部分が欠けるため、モデルはより複雑なパターンを学習することで汎化性能の向上が期待できる。
[注意機構を備えたロボット動作生成モデル](../model/SARNN.md)に適用すると、画像内の欠けた部分には注意が向かないことから、結果的に、動作予測に重要な空間的注意を探索（学習）することが可能になる。
ソースコードは[こちら](https://github.com/ogata-lab/eipl/blob/master/eipl/layer/GridMask.py)から。

[![grid_mask](img/grid_mask.png)](img/grid_mask.png)


