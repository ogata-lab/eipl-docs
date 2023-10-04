# Overview

Here we will describe a sequence of procedures from motion teaching to motion generation using [Open Manpulator](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/). The following five points will be covered:

!!! note
    Coming soon.

1. Motion Teaching

    In this step, the motion is taught to the Open Manipulator robot. Different methods can be used to teach the motion, such as using a leader-follower system or a joystick.
    
2. Data Collection

    After motion teaching, data collection is performed to capture the robot's sensor information, including joint angles, camera images, and other relevant data. This data is used to train the motion generation model.

3. Data Set Preparation

    Once the data is collected, it must be prepared in a suitable format for training the motion generation model. This includes organizing the data, performing any necessary preprocessing steps such as normalization or resizing, and dividing the data into training and test sets.

4. Model Training
    
    With the prepared data set, the motion generation model is trained using machine learning techniques. This typically involves feeding the input data, such as joint angles and camera images, into the model and adjusting the model parameters to minimize the prediction error. 

5. Motion Generation

    After the model is trained, it can be used to generate motion. By providing the desired input, such as target joint angles or desired end-effector positions, the model can generate appropriate motion commands for the robot. These motion commands can be sent to the robot's actuators, allowing it to perform the desired motions.



<!-- 
1. Hardware Configuration
2. ROS Environment
3. Data Collection
4. Generate Dataset
5. Model Training
6. Online Motion Generation
-->
