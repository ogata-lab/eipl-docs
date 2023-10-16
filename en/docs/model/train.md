

## Files
Use the programs in the [tutorial/SARNN](https://github.com/ogata-lab/eipl/tree/master/eipl/tutorials/airec/sarnn) folder of the EIPL repository to train SARNN. Each folder and each program has a specific role:

- **bin/train.py**: Program to load data, train and save models.
- **bin/test.py**: Program for offline inference of models using test data (images and joint angles) and visualization of inference results.
- **bin/test_pca_sarnn.py**: Program to visualize the internal state of the RNN using Principal Component Analysis.
- **libs/fullBPTT.py**: Backpropagation class for time series learning.
- **log**: Folder for storing weights, learning curves, and parameter information.
- **output**: Folder for storing inference results.


<!-- #################################################################################################### -->
----
## Trainig {#train}
The main program `train.py` is used to train SARNN. When the program is executed, the weights (pth) and Tensorboard log files are saved in the `log` folder. The program allows users to specify necessary training parameters such as model type, number of epochs, batch size, learning rate, and optimization method using command line arguments. It also uses the EarlyStopping library to determine when to stop training early and save weights when the test error is minimized. For a detailed explanation of how the program works, please refer to the comments in the [code] (https://github.com/ogata-lab/eipl/blob/master/eipl/tutorials/airec/sarnn/bin/train.py).


```bash 
$ cd eipl/tutorials/sarnn/
$ python3 ./bin/train.py
[INFO] Set tag = "20230521_1247_41"
================================
batch_size : 5
device : 0
epoch : 100000
heatmap_size : 0.1
img_loss : 0.1
joint_loss : 1.0
k_dim : 5
log_dir : log/
lr : 0.001
model : sarnn
optimizer : adam
pt_loss : 0.1
rec_dim : 50
stdev : 0.02
tag : "20230521_1247_41"
temperature : 0.0001
vmax : 1.0
vmin : 0.0
================================
12%|████          | 11504/100000 [14:46:53<114:10:44,  4.64s/it, train_loss=0.000251, test_loss=0.000316]
```


<!-- #################################################################################################### -->
----
## Learning Curves {#tensorboard}
You can check the training progress of the model using TensorBoard. By specifying the log folder where the weights are stored with the `logdir` argument, you can visualize the learning curve in your browser, as shown in the figure below. If there is a tendency for overfitting in the early stages of training, it may be due to anomalies in the training data or model, or in the initial weights (seeds). Countermeasures include checking the normalization range of the training data, reviewing the model structure, and retraining with different seed values. For specific instructions on how to use TensorBoard, please refer to the documentation linked [here] (https://www.tensorflow.org/tensorboard).


```bash
$ cd eipl/tutorials/sarnn/
$ tensorboard --logdir=./log/
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.12.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

![Learning_curve_using_tensorbaord](img/tensorboard.webp){: .center}

