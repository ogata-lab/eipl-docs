# Setup

<!-- ******************************** -->
----
## ROS
In this section, we use the rospy and rosbag packages to extract data from `rosbag` files. If you are creating datasets in an environment with ROS installed, the following procedure is not necessary. Please skip to the [next chapter](./dataset.md).


## pyenv
On the other hand, if you want to use rospy and other software on a PC without a ROS environment, you can use the [rospypi/simple](https://github.com/rospypi/simple) package. This package allows the use of binary packages such as rospy and tf2 without the need for a full ROS installation. Furthermore, since it is compatible with Linux, Windows, and MacOS, you can easily analyze the collected data in your own PC environment. To avoid conflicts with existing Python environments, it is recommended to create a virtual environment using venv. The following procedure outlines the steps to create an environment for the rospypi/simple library using venv.


```bash
$ python3 -m venv ~/.venv/rosbag
$ source ~/.venv/rosbag/bin/activate
$ pip install -U pip
$ pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
$ pip install matplotlib numpy opencv-python
```

!!! note
    
    The authors have not been able to confirm that the rospypi/simple library can handle all types of message data. In particular, custom ROS messages have not been tested. Therefore, if a program cannot be executed correctly in a virtual environment, it should be run in a ROS environment.