## Joint Angle Normalization {#joint_norm}

Since the robot's camera image consists of integer values representing 256 shades (0-255 = 8 bits), it is necessary to normalize the values to fit within a specified range, such as [0.0, 1.0], instead of the original range of [0, 255]. On the other hand, the possible range of robot joint angles can vary depending on factors such as joint structure, range of motion, and teaching task. The simplest normalization method is to normalize the joint angles based on the maximum and minimum values of the training data. However, with this approach, the normalization range is affected by joints with large movements, making it difficult to learn fine movements accurately. To solve this problem, the joint angles are normalized based on the maximum and minimum values of each individual joint, allowing fine movements to be emphasized and learned more effectively.

The figure below shows the results of joint angle normalization during object grasping. From left to right, the results show the normalization based on the maximum and minimum values of the collected raw data and training data (overall normalization), and the normalization based on the maximum and minimum values of each joint angle (joint normalization). In the case of overall normalization, the waveform after normalization shows minimal changes compared to the raw data, and the range of joint angles is simply converted to a scale from 0.0 to 1.0. However, when joint normalization is applied, both the coarse motion (e.g., represented by the gray waveform) and the fine motion (represented by the blue waveform) become more pronounced due to the normalization performed on each individual joint. This allows for more accurate learning of the robot's motions, surpassing the capabilities of global normalization alone.


[![joint_norm](img/joint_norm.png)](img/joint_norm.png)



!!! note
    Depending on the task, certain joints may have minimal or no motion. If joint normalization is applied to such joints, the resulting waveforms after normalization may be severely distorted, which can negatively affect the learning process. It is advisable to inspect the waveforms after normalization and manually adjust the normalization range if distorted joint waveforms are present. It is important to note that joint normalization is not suitable for inherently noisy data, such as torque and current values.

----
## Cosine Interpolation {#cos-interpolation}

In cases where the model is learning data represented as ON/OFF states, such as robot hand open/close commands or signals from a Position Sensitive Detector (PSD) sensor, it may be beneficial to apply smoothing beforehand to facilitate learning. The following example shows the effect of applying smoothing using cosine interpolation to the original data (represented by a blue square wave). The degree of smoothness can be adjusted by modifying the `step_size` argument.


[![interpolation](img/interpolation.png)](img/interpolation.png)
