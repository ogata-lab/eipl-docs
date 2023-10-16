# Introduction


EIPL (Embodied Intelligence with Deep Predictive Learning) is a library for robot motion generation using deep predictive learning developed at the  [Ogata Laboratory](https://ogata-lab.jp/), Waseda University. [Deep predictive learning](overview.md) is a method that enables flexible motion generation for unlearned environments and work goals by predicting the appropriate motion for the real world in real time based on past learning experience. In this study, we use the humanoid robot AIREC [AIREC (AIREC：AI-driven Robot for Embrace and Care)](https://airec-waseda.jp/en/toppage_en/) and [Open Manpulator](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/) as real robots, which enables systematic learning from model implementation to learning and real-time motion generation. In the future, newly developed motion generation models using EIPL will be published in the [Model Zoo](./zoo/overview.md). Below is an overview of each chapter.


1. [**Deep Predictive Learning**](overview/)

    This section explains the concept of deep predictive learning and outlines the three steps towards robot implementation: motion teaching, learning, and motion generation.
    
2. [**Set Up**](install/install-software)

    This section provides instructions on how to install EIPL and verify the program using pre-trained weights.

3. [**Motion Teaching**](teach/overview)

    This section describes the process of extracting data from ROSbag files and creating datasets. EIPL provides a sample dataset of object grasping motion using AIREC.

4. [**Teaching Model**](model/dataloader)

    Using the attention mechanism based motion generation model as an example, this section explains the implementation steps for training the model and performing inference.

5. [**Robot Simulator**](simulator/dataset)

    This section describes motion learning using a robot simulator (robosuite).

6. [**Real Robot Application**](robot/overview)

    This section provides a detailed explanation of the procedures involved in applying motion learning to real robot control using Open Manpulator.

7. [**Model Zoo**](zoo/overview)

    The motion generation models developed with EIPL will be gradually released in the ModelZoo.

8. [**Tips and Tricks**](tips/normalization/)

    This section provides valuable insights and tips on motion learning techniques.



----
**Acknowledgements**

This work was supported by JST Moonshot-type R&D Project JPMJMS2031. We would like to express our gratitude.
