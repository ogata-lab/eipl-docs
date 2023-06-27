# Deep Predictive Learning {#dpl}

## Overview {#dpl-overview}
Deep Predictive Learning (Deep Predictive Learning) is a robot motion generation method developed with reference to the free energy principle, which unifies and explains the various functions of the brain[@ito2022efficient]. A recurrent coupled neural network (RNN) model is trained to minimize the prediction error between the sensory-motor information at time (t) and the next time (t+1) using the time-series information of the robot's motion and sensation when the robot experiences motion in the real world using teleoperation, etc. At runtime, it is possible to predict the robot's near-future sensations and actions in real time from its sensory-motor information, and to execute actions such that the resulting prediction error is minimized by the attracting action of attractors in the RNN. The figure below shows a robot implementation of deep predictive learning, which consists of three steps: motion teaching, learning, and motion generation.


[![overview of deep predictive learning](img/dpl-overview.webp)](img/dpl-overview.webp)

      

----
## Motion Teaching {#dpl-teach}
In deep predictive learning, motion generation models are acquired in a data-driven manner using the robot's sensorimotor information (time-series data comprising motion and sensor information) as training data. Therefore, the training data must contain information about the interaction between the robot's body and the environment. In addition, since the quantity and quality of the training data influence model performance, high-quality demonstration data must be collected efficiently. 

In Phase 1, the training data is collected by performing the desired motions on the robot and recording motion information, such as joint angles, and sensor information, such as camera images, at a constant sampling rate. Typical teaching methods for robot motion include program description[@suzuki2021air], direct teaching[@ichiwara2022contact], and teleoperation[@ito2022efficient]. Although describing the robot's motion in advance using a robot programming language is simple, it may not be feasible due to the complexity of the description when a robot needs to perform a long-term motion. In contrast, training data without precise modeling and parameter adjustment can be obtained by teaching movements through human manipulation of the robot. Among them, remotely controlling the manipulator from the actual robot's perspective (Wizard of Oz[@yang2016repeatable]) is desirable because it can intuitively teach the robot human manipulation skills for a task. The operator interacts naturally with the environment by controlling the robot as if he were controlling his body. In addition, since the operator makes decisions about actions based on information obtained from sensor information, the acquired teaching dataset is expected to contain information necessary for motion learning and be effective in acquiring data for model training.


----
## Training {#dpl-train}
In deep predictive learning, the learning target is the time-series relationships between sensorimotor information in a system in which the environment and the body interact dynamically. The training data are not labeled with correct answers, and the model is trained to predict the robot state ($\hat i_t, \hat s_{t+1}$) in the next step using the current robot state as input ($i_t, s_t$). This autoregressive learning eliminates the need for the detailed design of a physical model of the environment, as required in conventional robotics. In addition, the model can be represented as a dynamic system integrating environment recognition and motion generation functions across multiple modalities.

The model consists of feature extraction and time series learning parts to learn the robot's sensorimotor information. The feature extraction part extracts features from sensor information values acquired by the robot, and the time series learning part learns sensorimotor information that integrates the extracted features and robot motion information ( e.g., joint angles and torque). Although each part is connected end-to-end ([CNNRNN](zoo/CNNRNN.md), [SARNN](model/SARNN.md)) or learned independently ([CAE-RNN](zoo/CAE-RNN.md)), the roles of each part are explicitly separated in this manual.


----
## Motion Generation {#dpl-execute}
When performing the task, RNN performs three processes sequentially: (1) Acquire sensor information from the robot, (2) predict the next state based on the sensor information, and (3) send control commands to the robot based on the predicted values. By performing the forward computation of the model at each step, RNN predicts the robot's state for the next time-step based on the context information and inputs it holds internally. The RNN output is then used as the target state to control each joint. By repeating the above process online, the RNN predicts the sensorimotor motion of the robot while sequentially changing the state of each neuron in the context layer. Based on the results of this prediction and the prediction error in the real environment, the robot generates motions that dynamically respond to the input.

Another advantage of using deep learning models for motion generation is the online motion generation speed. The proposed framework comprises lightweight models, and the computational time and cost required for motion generation are low. Moreover, implementing each of the previous functions as components makes it possible to easily reuse the implemented system when tasks or robot hardware changes, or devices are added[@kanamura2021development].
