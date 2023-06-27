# Quick Start

In this section, we run a test program to verify the proper installation of the EIPL environment. We will use the pre-trained weights and the motion generation model with spatial attention mechanism (SARNN: Spatial Attention with Recurrent Neural Network). For specific details on model training methods and model specifications, please refer to the following chapters.

## Inference
To perform inference using SARNN and the pre-trained weights, execute the `test.py` file in the tutorial folder. The resulting inferences are saved in the output folder. Specifying the `--pretrained` argument will automatically download the pre-trained weights and sample data.


``` bash linenums="1"
$ cd eipl/tutorials/sarnn
$ python3 ./bin/test.py --pretrained
$ ls ./output/
SARNN_20230514_2312_17_4_1.0.gif
```

## Results
The figure below shows the inference results. The blue dots in the figure represent the attention points extracted by the Convolutional Neural Network (CNN), while the red dots indicate the attention points predicted by the Recurrent Neural Network (RNN). This visualization shows the prediction of joint angles with a focus on the robot hand and the grasped object.

![results_of_SARNN](img/sarnn-rt_4.webp){: .center}


## Help
If an error occurs, there are three possible causes:

1. **Installation error**

    To ensure proper installation, use the "pip freeze" command to verify that the libraries are installed correctly. If the library is installed, its version information will be displayed. If not, it is possible that the package was not installed properly, so please [check](./install-software.md#pip_install) the installation procedure again.

        pip freeze | grep eipl


2. **Download error**

    If you have problems downloading the [sample data](https://dl.dropboxusercontent.com/s/5gz1j4uzpzhnttt/grasp_bottle.tar) or the [pre-trained weights file](https://dl.dropboxusercontent.com/s/o29j0kiqwtqlk9v/pretrained.tar) due to a proxy or other reason, you can manually download the weights file and the data set. Save them to the `~/.eipl/` folder and then extract the files.

        $ cd ~/
        $ mkdir -p .eipl/airec/
        $ cd .eipl/airec/
        $ # copy grasp_bottle.tar and pretrained.tar to ~/.eipl/airec/ directory
        $ tar xvf grasp_bottle.tar && tar xvf pretrained.tar
        $ ls grasp_bottle/*
        grasp_bottle/joint_bounds.npy
        ...
        $ ls pretrained/*
        pretrained/CAEBN:
        args.json  model.pth
        ...


3. **Drawing error**

    If you see the following error message after running the program, it may indicate an error in generating the animation file. In such cases, modifying the code at the end of test.py will solve the problem.
    
        File "/usr/lib/python3/dist-packages/matplotlib/animation.py", line 410, in cleanup
            raise subprocess.CalledProcessError(
        subprocess.CalledProcessError: Command '['ffmpeg', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '720x300', '-pix_fmt', 'rgba', '-r', '52.63157894736842', '-loglevel', 'error', '-i', 'pipe:', '-vcodec', 'h264', '-pix_fmt', 'yuv420p', '-y', './output/CAE-RNN-RT_20230510_0134_03_0_0.8.gif']' returned non-zero exit status 1.


    First, use the `apt` command to install imagemagick and ffmpeg.

        $ sudo apt install imagemagick
        $ sudo apt install ffmpeg
    
    Next, make the following changes to the code at the bottom of `test.py`:

        # Using imagemagick
        ani.save( './output/SARNN_{}_{}_{}.gif'.format(params['tag'], idx, args.input_param), writer="imagemagick") 
        
        # Using ffmpeg
        ani.save( './output/SARNN_{}_{}_{}.gif'.format(params['tag'], idx, args.input_param), writer="ffmpeg") 
