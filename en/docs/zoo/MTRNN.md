# Overview

MTRNN is a type of RNN consisting of a hierarchical group of neurons with different firing rates[@yamashita2008emergence]. It consists of three layers: an input-output (IO) layer and context layers (Cf and Cs layers) with different firing rates (time constants), each with recursive inputs. The time constants increase from the Cf layer to the Cs layer, resulting in slower response speeds to the input. The input information is then passed through the Cf and Cs layers to the output layer. There is no direct connection between the IO and Cs layers, and their interaction occurs through the Cf layer. The MTRNN allows the robot to learn behaviors, where the Cf layer represents behavioral primitives and the Cs layer represents learning the combination of these primitives. Compared to LSTM, MTRNN is more interpretable and is widely used in [our lab](https://ogata-lab.jp/).



![MTRNN](img/mtrnn/mtrnn.webp){: .center}


::: MTRNN.MTRNNCell
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: MTRNN.BasicMTRNN
    handler: python
    options:
      show_root_heading: true
      show_source: true
