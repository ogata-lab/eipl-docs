# Overview

EPIL officially supports Linux (Ubuntu 20.04) and Python 3.8.
Pytorch is used as the deep learning framework and it is recommended to install the [latest version of pytorch](https://pytorch.org/get-started/locally/).
In particular, Pytorch 2.0 and above can perform training of large models faster because it is precompiled, which improves training speed and reduces GPU memory usage.
Note that CUDA and Nvidia drivers must be installed according to the [version](https://pytorch.org/get-started/previous-versions/) of Pytorch used.

----
## Files
This library is composed of the following:

- **data**: A sample data downloader and a Dataloader for model training are implemented.
- **layer**: A layered model ([Hierarchical RNNs](../zoo/MTRNN.md), [spatial attention mechanisms](../model/SARNN.md#spatial_softmax), etc.) are implemented.
- **model**: Multiple motion generation models are implemented, and inputs support joint angles (arbitrary degrees of freedom) and color images (128x128 pixels).
- **test**: Test programs.
- **utils**: Functions for normalization, visualization, arguments processing, etc.

----
## Install from pip {#pip_install}

Clone the EIPL repository from Github and install the environment using the pip command.

```bash linenums="1"
mkdir ~/work/
cd ~/work/
git clone https://github.com/ogata-lab/eipl.git
cd eipl
pip install -r requirements.txt
pip install -e .
```

----
## docker

!!! Note
    Coming soon.