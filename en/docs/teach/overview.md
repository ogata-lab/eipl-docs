# Overview

This section provides instructions on how to create a dataset for deep predictive learning using robot sensor data collected by the ROS system. For better understanding, it is recommended to download [the collected data and scripts (1.3GB)](https://dl.dropboxusercontent.com/s/90wkfttf9w0bz0t/rosbag.tar) and follow the instructions to run them.



<!-- ******************************** -->
----
## Experimental Task {#task}
AIREC (AI-driven Robot for Embrace and Care), a humanoid robot developed by [Tokyo Robotics](https://robotics.tokyo/), is used to teach object grasping. The figure below shows an overview of the task. The generalization performance is evaluated by comparing the object grasping experience at the teaching positions (three circled points) with the untaught positions (two points) shown in the figure. Training data are collected four times for each teaching position, for a total of 12 data points. Test data is collected once for each of the five positions, including the untaught positions, for a total of five data


![task_overview](img/teaching.webp){: .center}


<!-- ******************************** -->
----
## Motion Teaching {#teaching}
AIREC is a robotic system that enables bilateral teleoperation, as shown below. The operator can teach a multi-degree of freedom robot more intuitively by instructing its motion based on the visual image of the robot displayed on the monitor and receiving force feedback from the robot. During the task teaching process using the teleoperation device, sensor information such as joint angles, camera images, torque information, etc. is stored in the "rosbag" format and a dataset is created for the deep predictive learning model in the following sections.

Note that it is possible to teach motion to a robot without using such specialized equipment.  [The Real Robot Application section](../robot/overview.md) describes two motion teaching methods using OpenManipulator: the leader-follower system and joystick control.


<html lang="ja">
<head>
  <link rel="stylesheet" href="index.css">
</head>
<body>
  <div class="wrap">
    <iframe class="youtube" width="640" height="360" src="https://www.youtube.com/embed/ivksUcWIK4g" title="Bilateral teleoperation of a humanoid robot Dry-AIREC" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div>
</body>
</html>

