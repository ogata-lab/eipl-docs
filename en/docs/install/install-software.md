# Overview

EPIL officially supports Linux (Ubuntu 20.04), Python 3.8 and PyTorch with the [latest version](https://pytorch.org/get-started/locally/). In particular, PyTorch 2.0 allows faster training of large models through precompilation, resulting in improved training speed and reduced GPU memory usage. Please make sure that CUDA and Nvidia drivers are installed according to the PyTorch [version](https://pytorch.org/get-started/previous-versions/) you are using.

----
## Software Files
This library consists of the following components:

- **data**: Implements a sample dataset downloader and a Dataloader for model training.
- **layer**: Implements layered models such as([Hierarchical RNNs](../zoo/MTRNN.md) and [spatial attention mechanisms](../model/SARNN.md#spatial_softmax), etc.
- **model**: Implements multiple motion generation models, with support for inputs including joint angles (with arbitrary degrees of freedom) and color images (128x128 pixels).
- **test**: Contains test programs.
- **utils**: Provides functions for normalization, visualization, argument processing, etc.

----
## Install via pip {#pip_install}

To set up the environment, clone the EIPL repository from GitHub and install it using the pip command.

```bash linenums="1"
mkdir ~/work/
cd ~/work/
git clone https://github.com/ogata-lab/eipl.git
cd eipl
pip install -r requirements.txt
pip install -e .
```
