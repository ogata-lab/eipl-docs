# Setup

<!-- ******************************** -->
----
### ROS
Here, we use the rospy and rosbag packages to extract data from `rosbag`.
If you are generating data sets in a ROS installed environment, the following process is not necessary, so please go to the [next chapter](./dataset.md).

### pyenv
On the other hand, one approach to using rospy and other software on a PC without ROS installed is [rospypi/simple](https://github.com/rospypi/simple).
This package enables the use of binary packages such as rospy and tf2 without installing ROS.
Furthermore, since it is compatible with Linux, Windows, and MacOS, the collected data can be easily analyzed on one's own PC environment.
In order to prevent conflicts with existing python environments, it is recommended to create a virtual environment using venv.
The following is the procedure for creating an environment for rospypi/simple library using venv.

```bash
$ python3 -m venv ~/.venv/rosbag
$ source ~/.venv/rosbag/bin/activate
$ pip install -U pip
$ pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
$ pip install matplotlib numpy opencv-python
```

!!! note
    
    The authors have not been able to verify that the rospypi/simple library can handle all message data.
    Especially custom ROS messages have not been tested, so if a program cannot be executed correctly in a virtual environment, it should be executed in a ROS environment.