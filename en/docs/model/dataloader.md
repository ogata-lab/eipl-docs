


EIPL provides a `MultimodalDataset` class for learning robot motions, which inherits from the Dataset class. This class returns a pair of input data, `x_data`, and the corresponding true value, `y_data`, for each epoch. The input data, `x_data`, consists of pairs of images and joint angles, and data augmentation is applied during each epoch. The input images are randomly adjusted for brightness, contrast, etc. to improve robustness to changes in lighting conditions, while Gaussian noise is added to the input joint angles to improve robustness to position errors. On the other hand, no noise is added to the original data. The model learns to handle noiseless situations (internal representation) from input data mixed with noise, allowing robust motion generation even in the presence of real-world noise during inference.

The following source code shows how to use the `MultimodalDataset` class with an example of an [object grasping task](../teach/overview.md) collected by AIREC. By providing 5-dimensional time series image data [number of data, time series length, channel, height, width] and 3-dimensional time series joint angle data [number of data, time series length, number of joints] to the `MultimodalDataset` class, data augmentation and other operations are performed automatically. Note that the `SampleDownloader`, which is used to download the sample dataset, is not mandatory. You can use functions like `numpy.load` or others to load your own datasets directly.



```python title="How to use dataloader" linenums="1"
from eipl.data import SampleDownloader, MultimodalDataset

# Download and normalize sample data
grasp_data = SampleDownloader("airec", "grasp_bottle", img_format="CHW")
images, joints = grasp_data.load_norm_data("train", vmin=0.1, vmax=0.9)

# Give the image and joint angles to the Dataset class
multi_dataset = MultimodalDataset(images, joints)

# Return input/true data as return value.
x_data, y_data = multi_dataset[1]
```

The following figure shows the robot camera images returned by the `MultimodalDataset` class. From left to right, the images show the original image, the image with noise, and the robot joint angles. Random noise is added to the image at each epoch, allowing the model to learn from a variety of visual situations. The black dotted lines represent the original joint angles, while the colored lines represent the joint angles with Gaussian noise.


![dataset](img/vis_dataset.webp){: .center}


!!! note
    
    If you are unable to obtain the dataset due to a proxy or any other reason, you can manually download the dataset from [here](https://dl.dropboxusercontent.com/s/5gz1j4uzpzhnttt/grasp_bottle.tar) and save it in the ~/.eipl/ folder.

        ```bash            
        $ cd ~/
        $ mkdir -p .eipl/airec/
        $ cd .eipl/airec/
        $ # copy grasp_bottle.tar to ~/.eipl/airec/ directory
        $ tar xvf grasp_bottle.tar
        $ ls grasp_bottle/*
        grasp_bottle/joint_bounds.npy
        ```


<!-- #################################################################################################### -->
---- 
::: dataloader.MultimodalDataset
    handler: python
    options:
      show_root_heading: true
      show_source: true
