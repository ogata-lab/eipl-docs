# Quick Start

In this section, a test program will be run using the pre-trained weights and the motion generation model with spatial attention mechanism (SARNN: Spatial Attention with Recurrent Neural Network) to verify that the EIPL environment has been properly installed. 
Please refer to the next and subsequent chapters for specific model training methods and model details.

## Inference
The following shows how to infer SARNN using the pre-trained weights and running `test.py` in the tutorial folder will save the inference results in the output folder.
At this time, by specifying the --pretrained argument, the pre-trained weights and sample data are automatically downloaded.

``` bash linenums="1"
$ cd eipl/tutorials/sarnn
$ python3 ./bin/test.py --pretrained
$ ls ./output/
SARNN_20230514_2312_17_4_1.0.gif
```

## Results
The following figure shows the result of inference. The blue dots in the figure are the points of attention extracted from the CNN (Convolutional Neural Network),
and the red dots are the points of attention predicted by the RNN (Recurrent Neural Network),
indicating that the joint angles are predicted while focusing on the robot hand and grasped object.

![results_of_SARNN](img/sarnn-rt_4.webp){: .center}


## HELP
If an error occurs, there are three possible causes:

1. **Installation error**

    Since the libraries may not be properly installed, use the `pip freeze` command to verify that they are installed. If the library is installed, its version information will be displayed. If not, the package may not have been installed, so please double check the [installation procedure](./install-software.md#pip_install).

        pip freeze | grep eipl


2. **Download error**

    If you are unable to perform inference using sample data or trained models due to a proxy or other reason, manually download the [weights file](https://dl.dropboxusercontent.com/s/o29j0kiqwtqlk9v/pretrained.tar) and [dataset](https://dl.dropboxusercontent.com/s/5gz1j4uzpzhnttt/grasp_bottle.tar), save them in the `~/.eipl/` folder, and then extract them.
        
        $ cd ~/
        $ mkdir .eipl
        $ cd .eipl
        $ # copy grasp_bottle.tar and pretrained.tar to ~/.eipl/ directory
        $ tar xvf grasp_bottle.tar && tar xvf pretrained.tar
        $ ls grasp_bottle/*
        grasp_bottle/joint_bounds.npy
        ...
        $ ls pretrained/*
        pretrained/CAEBN:
        args.json  model.pth
        ...


3. **Drawing error**

    If the following error message appears after program execution, the animation file may have failed to be generated.
    In this case, changing the WRITER when drawing the animation will solve the problem.
    
        File "/usr/lib/python3/dist-packages/matplotlib/animation.py", line 410, in cleanup
            raise subprocess.CalledProcessError(
        subprocess.CalledProcessError: Command '['ffmpeg', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '720x300', '-pix_fmt', 'rgba', '-r', '52.63157894736842', '-loglevel', 'error', '-i', 'pipe:', '-vcodec', 'h264', '-pix_fmt', 'yuv420p', '-y', './output/CAE-RNN-RT_20230510_0134_03_0_0.8.gif']' returned non-zero exit status 1.


    First, install imagemagick and ffmpeg using the `apt` command.
        
        $ sudo apt install imagemagick
        $ sudo apt install ffmpeg
    
    Next, edit the code at the bottom of `test.py` as follows:

        # Using imagemagick
        ani.save( './output/SARNN_{}_{}_{}.gif'.format(params['tag'], idx, args.input_param), writer="imagemagick") 
        
        # Using ffmpeg
        ani.save( './output/SARNN_{}_{}_{}.gif'.format(params['tag'], idx, args.input_param), writer="ffmpeg") 
