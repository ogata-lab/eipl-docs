# Introduction

EIPL (Embodied Intelligence with Deep Predictive Learning) is a library for robot motion generation using deep predictive learning developed at [Tetsuya Ogata Laboratory](https://ogata-lab.jp/), Waseda University. [Deep predictive learning](overview.md) is a method that enables flexible motion generation for unlearned environments and work targets by predicting in real time the appropriate motion for the real world based on past learning experience. In this study, using the smart robot [AIREC (AIREC：AI-driven Robot for Embrace and Care)](https://airec-waseda.jp/en/toppage_en/) and[Open Manpulator](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/) as real robots, it is possible to learn systematically from model implementation to learning and real-time motion generation using collected visual and joint angle data. The EIPL system enables users to learn systematically from model implementation to learning and real-time motion generation using the collected visual and joint angle data. In the future, newly developed motion generation models using EIPL will be released on [Model Zoo](./zoo/overview.md). The following is an overview of each chapter.



1. [**Deep Predictive Learning**](overview/)

    The concept of deep predictive learning and the three steps toward robot implementation, motion teaching, learning, and motion generation, are described.
    
2. [**Setup**](install/install-software)

    Describes how to install EIPL and check the program using pre-trained weights.

3. [**Motion Teaching**](teach/overview)

    Describes how to extract data from ROSbag data and create dataset. EIPL provides an object grasping motion data set using AIREC as sample data.

4. [**Train Model**](model/dataloader)

    Using the motion generation model with attention mechanism as an example, describes a series of implementation methods from model training to inference.

5. [**Real Robot Application**](robot/overview)

    Describes a series of procedures from motion teaching to real robot control using Open Manpulator.

6. [**ModelZoo**](zoo/overview)

    The motion generation models developed using EIPL will be released in a phased manner.

7. [**Tips**](tips/normalization/)

    Describes the know-how of motion learning.




----
**Acknowledgements**

This work was supported by JST Moonshot-type R&D Project JPMJMS2031. We would like to express our gratitude.
