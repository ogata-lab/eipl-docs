
## transforms.RandomAffine
[transforms.RandomAffine](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAffine.html) is a function for applying a random affine transformation to an image. Affine transforms can transform an image by translating, rotating, scaling, or distorting it. The figure below shows the result of translating an image vertically and horizontally. When affine transforms are used for training AutoEncoder, the position information of objects is expressed (extracted) as image features, so that even unlearned positions can be reconstructed appropriately.


[![random_affine](img/random_affine.png)](img/random_affine.png)


----
## transforms.RandomVerticalFlip
[transforms.RandomVerticalFlip](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomVerticalFlip.html is a function that randomly flips the input image upside down to increase data diversity.

[![vertical_flip](img/vertical_flip.png)](img/vertical_flip.png)


----
## transforms.RandomHorizontalFlip
[transforms.RandomHorizontalFlip](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html) is a function that randomly flips the input image left and right, and can be combined with `RandomVerticalFlip` to improve the generalization performance of the model.


[![horizontal_flip](img/horizontal_flip.png)](img/horizontal_flip.png)


----
## transforms.ColorJitter
[transforms.ColorJitter](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html) is a function that performs a random color transformation on an input image, allowing the brightness, contrast, saturation, and hue of the image to be changed, as shown in the figure below.

[![color_jitter](img/color_jitter.png)](img/color_jitter.png)


----
## GridMask
GridMask is a method to increase the diversity of training data by hiding parts of the image using a grid-like pattern[@chen2020gridmask].
As shown in the figure below, the model is expected to improve generalization performance by learning image data with some part missing.
When applied to a [SARNN model](../model/SARNN.md),
the missing parts of the image will not attract attention, and as a result, the model can explore (learn) spatial attention that is important for motion prediction.
Source code is available [here](https://github.com/ogata-lab/eipl/blob/master/eipl/layer/GridMask.py).

[![grid_mask](img/grid_mask.png)](img/grid_mask.png)


