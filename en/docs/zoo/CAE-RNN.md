# CAE-RNN {#cae-rnn}

CAE-RNN is a motion generation model consisting of an image feature extraction part and a time series learning part to learn the robot's sensory-motor information [@ito2022efficient, @yang2016repeatable].
The following figure shows the network structure of the CAE-RNN model, which consists of a Convolutional Auto-Encoder (CAE) that extracts image features from the robot's visual information, and a Recurrent Neural Network (RNN) that learns the time series information of robot's joint angles and image features.
CAE-RNN features independent training of the image feature extraction part and the time series learning part, which are trained in the order of CAE and RNN.
By learning a variety of sensory-motor information, it is possible to extract image features such as the position and shape of flexible objects that are conventionally difficult to recognize, and to learn and generate corresponding motions.
This section describes a series of processes from the [CAE](#cae) and [RNN](#rnn) model implementation, training, inference, [internal representation analysis](#rnn_pca).

![CAE-RNN](img/cae-rnn/cae-rnn.png){: .center}


<!-- #################################################################################################### -->
----
## CAE {#cae}
### Overview {#cae_overview}
Since visual images are high-dimensional information compared to the robot's motion information, it is necessary to align the dimensions of each modal in order to properly learn sensory-motor information. 
Furthermore, in order to learn the relationship between the position and motion of the object, it is necessary to extract low-dimensional image features (e.g., position, color, shape, etc.) of the object or robot's body from the high-dimensional visual image.
Therefore, Convolutional Auto-Encoder (CAE) is used to extract image features.
The following figure highlights only the CAE network structure in CAE-RNN, which consists of an Encoder that extracts image features from the robot's visual information ($i_t$) and a Decoder that reconstructs the image ($\hat i_t$) from the image features.
By updating the weights of each layer to minizize the error between input and output values, the layer with the fewest number of neurons (bottleneck layer) in the middle layer is able to extract an abstract representation of the input information. 


![Network structure of CAE](img/cae-rnn/cae.png){: .center}

<!-- #################################################################################################### -->
----      
### Files {#cae_files}
The programs and folders used in CAE are as follows:

- **bin/train.py**: Programs to load data, train, and save models.
- **bin/test.py**: Program to perform off-line inference of models using test data (images and joint angles) and visualize inference results.
- **bin/extract.py**：Program to calculate and store the image features extracted by the CAE and the upper and lower limits for normalization.
- **libs/trainer.py**：Back propagation class for CAE.
- **log**: Folder to store weights, learning curves, and parameter information.
- **output**: Save inference results.
- **data**：Store RNN training data (joint angles, image features, normalization information, etc.).



<!-- #################################################################################################### -->
----
### CAE Model  {#cae_model}
CAE consists of a convolution layer, transposed convolution layer, and a linear layer.
By using the Convolution layer (CNN) to extract image features, CAE can handle high-dimensional information with fewer parameters compared to AutoEncoder [@hinton2006reducing], which consists of only a Linear layer. Furthermore, CNN can extract a variety of image features by convolving with shifting filters. The Pooling layer, which is generally applied after CNN, is often used in image recognition and other fields to compress the dimensionality of input data. However, while position invariance and information compression can be achieved simultaneously, there is a problem that information on the spatial structure of the image is lost [@sabour2017dynamic]. Since spatial position information of manipulated objects and robot hands is essential for robot motion generation, dimensional compression is performed using the convolution application interval (stride) of the CNN filter instead of the Pooling Layer.

The following is a program of the CAE model, which can extract image features of the dimension specified by `feat_dim` from a 128x128 pixel color image.
This model is a simple network structure to understand the outline and implementation of CAE.

```python title="<a href=https://github.com/ogata-lab/eipl/blob/master/eipl/model/CAE.py>[SOURCE] BasicCAE.py</a>" linenums="1"
class BasicCAE(nn.Module):
    def __init__(self,
                 feat_dim=10):
        super(BasicCAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  64, 3, 2, 1), nn.Tanh(),
            nn.Conv2d(64, 32, 3, 2, 1), nn.Tanh(),
            nn.Conv2d(32, 16, 3, 2, 1), nn.Tanh(),
            nn.Conv2d(16, 12, 3, 2, 1), nn.Tanh(),
            nn.Conv2d(12, 8,  3, 2, 1), nn.Tanh(),
            nn.Flatten(),
            nn.Linear(8*4*4, 50),   nn.Tanh(),
            nn.Linear(50, feat_dim),nn.Tanh()
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim,  50),   nn.Tanh(),
            nn.Linear(50, 8*4*4),       nn.Tanh(),
            nn.Unflatten(1, (8,4,4)), 
            nn.ConvTranspose2d(8, 12, 3, 2, padding=1, output_padding=1), nn.Tanh(),
            nn.ConvTranspose2d(12,16, 3, 2, padding=1, output_padding=1), nn.Tanh(),
            nn.ConvTranspose2d(16,32, 3, 2, padding=1, output_padding=1), nn.Tanh(),
            nn.ConvTranspose2d(32,64, 3, 2, padding=1, output_padding=1), nn.Tanh(),
            nn.ConvTranspose2d(64, 3, 3, 2, padding=1, output_padding=1), nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder( self.encoder(x) )
```

By using the ReLU function and Batch Normalization [@ioffe2015batch], it is possible to improve the expressiveness of each layer, prevent gradient loss, and furthermore make learning more efficient and stable.
In this library, CAE models using Batch Normalization have already been implemented and can be loaded as follows.
The difference between `BasicCAENE` and `CAEBN` is the structure of the model (parameter size), see [source code](https://github.com/ogata-lab/eipl/blob/master/eipl/model/CAEBN.py) for details.
Note that the input format of the implemented model is a color image of 128x128 pixels; if you want to input any other image size, you need to modify the parameters.


```python
from eipl.model import BasicCAENE, CAEBN
```


<!-- #################################################################################################### -->
----
### Back Propagation {#cae_bp}
In the CAE learning process, input camera images of the robot ($i_t$) and generate the reconstructed images ($\hat i_t$).
Next, the parameters of the model are updated using the back propagation method [@rumelhart1986learning] to minimize the error between the input and reconstructed images.
In lines 45-52, the batch size image $xi$ is input to the model to obtain the reconstructed image $yi_hat$.
Then, the mean square error `nn.MSELoss` between the reconstructed image and the true value $yi$ is calculated, and error propagation is performed based on the error value `loss`.
This autoregressive learning eliminates the need for detailed model design for images, which is required in conventional robotics.
Note that in order to extract image features that are robust against a variety of real-world noise, [data extension](../tips/augmentation.md) is used to train the model on images with randomly varying brightness, contrast, and position.


```python title="<a href=https://github.com/ogata-lab/eipl/blob/master/eipl/tutorials/cae/libs/trainer.py>[SOURCE] trainer.py</a>" linenums="1"
class Trainer:
    def __init__(self,
                model,
                optimizer,
                device='cpu'):

        self.device = device
        self.optimizer = optimizer        
        self.model = model.to(self.device)

    def save(self, epoch, loss, savename):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'train_loss': loss[0],
                    'test_loss': loss[1],
                    }, savename)

    def process_epoch(self, data, training=True):
        
        if not training:
            self.model.eval()

        total_loss = 0.0
        for n_batch, (xi, yi) in enumerate(data):
            xi = xi.to(self.device)
            yi = yi.to(self.device)

            yi_hat = self.model(xi)
            loss = nn.MSELoss()(yi_hat, yi)
            total_loss += loss.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / n_batch
```



<!-- #################################################################################################### -->
----
### Training {cae_train}
We will use `Model`, `Trainer Class` and the already implemented main program `train.py` to train CAE.
When the program is run, a folder name (e.g. 20230427_1316_29) is created in the `log` folder indicating the execution date and time.
The folder will contain the trained weights (pth) and the TensorBoard log file.
The program can use command line arguments to specify parameters necessary for training, such as model type, number of epochs, batch size, training rate, and optimization method.
It also uses the EarlyStopping library to determine when to end training early as well as to save weights when the test error is minimized (`save_ckpt=True`).
Please [see](https://github.com/ogata-lab/eipl/blob/master/eipl/tutorials/cae/bin/train.py) the comments in the code for a detailed description of how the program works.


```bash
$ cd eipl/tutorials/cae/
$ python3 train.py
[INFO] Set tag = 20230427_1316_29
================================
batch_size : 128
device : 0
epoch : 100000
feat_dim : 10
log_dir : log/
lr : 0.001
model : CAE
optimizer : adam
stdev : 0.02
tag : 20230427_1316_29
vmax : 1.0
vmin : 0.0
================================
0%|               | 11/100000 [00:40<101:55:18,  3.67s/it, train_loss=0.0491, test_loss=0.0454]
```



<!-- #################################################################################################### -->
----
### Inference {cae_inference}
Check that CAE has been properly trained using the test program `test.py`.
The argument `filename` is the path of the trained weights file and `idx` is the index of the data to be visualized.
The lower (top) figure shows the inference results of the `CAEBN` model using this program, with the input image on the left and the reconstructed image on the right.
Since the robot hand and the grasping object in the "unlearned position" are reconstructed, which is important for generating robot motion, it can be assumed that the image features represent information such as the object's position and shape.
The lower figure (bottom) is also an example of failure, showing that the object is not adequately predicted by the `Basic CAE` model with a simple network structure.
In this case, it is necessary to adjust the method of the optimization algorithm, the learning rate, the loss function, and the structure of the model.


```bash
$ cd eipl/tutorials/cae/
$ python3 test.py --filename ./log/20230424_1107_01/CAEBN.pth --idx 4
$ ls output/
CAEBN_20230424_1107_01_4.gif
```

![Reconstructed image using CAEBN](img/cae-rnn/basic_cae_inference.webp){: .center}

![Reconstructed image using BasicCAE](img/cae-rnn/caebn_inference.webp){: .center}



<!-- #################################################################################################### -->
---
### Extract image features {cae_extract_feat}
Extract image features of CAE as a preparation for time series learning of image features and robot joint angles with RNN.
Executing the following program, image features and joint angles of training and test data are stored in the `data` folder in npy format.
At this time, confirm that the number of data and time series length of the extracted image features and joint angles are the same.
The reason for storing the joint angles again is to make it easier to load the dataset when training RNN.

```bash
$ cd eipl/tutorials/cae/
$ python3 extract.py ./log/20230424_1107_01/CAEBN.pth
[INFO] train data
==================================================
Shape of joints angle: torch.Size([12, 187, 8])
Shape of image feature: (12, 187, 10)
==================================================

[INFO] test data
==================================================
Shape of joints angle: torch.Size([5, 187, 8])
Shape of image feature: (5, 187, 10)
==================================================

$ ls ./data/*
data/test:
features.npy  joints.npy

data/train:
features.npy  joints.npy
```

The following code is part of the source code of `extract.py`, which extracts and saves image features.
In the fourth line, the Encoder process of CAE is performed and the extracted low-dimensional image features are returned as the return value.
The image features extracted by CAE are normalized to within the range specified by the user, and then used for training RNN.
When `tanh` is used as the activation function of the model, the upper and lower bounds of the image features (`feat_bounds`) are constant (-1.0 to 1.0).
However, CAEBN uses `ReLU` for the activation function, so the upper and lower bounds of the image features are undetermined.
Therefore, in line 25, the upper and lower bounds of the image features are determined by calculating the maximum and minimum values from the extracted image features of the training and test data.


```python title="<a href=https://github.com/ogata-lab/eipl/blob/master/eipl/tutorials/cae/bin/extract.py>[SOURCE] extract.py</a>" linenums="1" hl_lines="4 25"
    # extract image feature
    feature_list = []
    for i in range(N):
        _features = model.encoder(images[i])
        feature_list.append( tensor2numpy(_features) )

    features = np.array(feature_list)
    np.save('./data/joint_bounds.npy', joint_bounds )
    np.save('./data/{}/features.npy'.format(data_type), features )
    np.save('./data/{}/joints.npy'.format(data_type), joints )
    
    print_info('{} data'.format(data_type))
    print("==================================================")
    print('Shape of joints angle:',  joints.shape)
    print('Shape of image feature:', features.shape)
    print("==================================================")
    print()

# save features minmax bounds
feat_list = []
for data_type in ['train', 'test']:
    feat_list.append( np.load('./data/{}/features.npy'.format(data_type) ) )

feat = np.vstack(feat_list)
feat_minmax = np.array( [feat.min(), feat.max()] )
np.save('./data/feat_bounds.npy', feat_minmax )
```



<!-- #################################################################################################### -->
----
## RNN {#rnn}

### Overview {#rnn_overview}
A Recurrent Neural Network (RNN) is used to integrate and learn the robot's sensory-motor information.
The following figure highlights only the network structure of RNN among CAE-RNNs, which inputs image features ($f_t$) and joint angles ($a_t$) at time `t` and predicts them at the next time `t+1`.

![Network structure of RNN](img/cae-rnn/rnn.png){: .center}


<!-- #################################################################################################### -->
----
### Files {#rnn_files}
The programs and folders used in RNN are as follows:

- **bin/train.py**: Programs to load data, train, and save models.
- **bin/test.py**: Program to perform off-line inference of models using test data (images and joint angles) and visualize inference results.
- **bin/test_pca_cnnrnn.py**: Program to visualize the internal state of RNN using Principal Component Analysis.
- **libs/fullBPTT.py**: Back propagation class for time series learning.
- **bin/rt_predict.py**: Program that integrates trained CAE and RNN model to predict motor command based on images and joint angles.
- **libs/dataloader.py**: DataLoader for RNN, returning image features and joint angles.
- **log**: Folder to store weights, learning curves, and parameter information.
- **output**: Save inference results.


<!-- #################################################################################################### -->
----
### RNN Model  {#rnn_model}
RNN is a neural network that can learn and infer time-series data, and it can perform time-series prediction by sequentially changing states based on input values.
However, Vanilla RNN is prone to gradient loss during bask propagation, to solve this problem,
Long Short-Term Memory (LSTM) and [Multiple Timescales RNN (MTRNN)](../zoo/MTRNN.md) have been proposed.

Here, we describe a method for learning integrated sensory-motor information of a robot using LSTM.
LSTM has three gates (input gate, forget gate, and output gate), each with its own weight and bias.
The $h_{t-1}$ gate learns detailed changes in the time series as short-term memory, and the $c_{t-1}$ gate learns features of the entire time series as long-term memory, allowing retention and forgetting of past information through each gate.
The following shows an example of implementation. Input value $x$, which is a combination of low-dimensional image features and robot joint angles extracted by CAE in advance, is input to LSTM.
LSTM then outputs the predicted value $\hat y$ of the image features and robot joint angles at the next time based on the internal state.


```python title="<a href=https://github.com/ogata-lab/eipl/blob/master/eipl/model/BasicRNN.py>[SOURCE] BasicRNN.py</a>" title="BasicRNN.py" linenums="1"
class BasicLSTM(nn.Module):
    def __init__(self,
                 in_dim,
                 rec_dim,
                 out_dim,
                 activation='tanh'):
        super(BasicLSTM, self).__init__()
        
        if isinstance(activation, str):
            activation = get_activation_fn(activation)

        self.rnn = nn.LSTMCell(in_dim, rec_dim )
        self.rnn_out = nn.Sequential(
            nn.Linear(rec_dim, out_dim),
            activation
        )  
    
    def forward(self, x, state=None):
        rnn_hid = self.rnn(x, state)
        y_hat   = self.rnn_out(rnn_hid[0])

        return y_hat, rnn_hid
```



<!-- #################################################################################################### -->
----
### Backpropagation Through Time {#rnn_bptt}
Backpropagation Through Time (BPTT) is used as the error back propagation algorithm for time series learning.
The details of BPTT have already been described in SARNN, please refer to [here](../../model/SARNN#bptt).


```python title="<a href=https://github.com/ogata-lab/eipl/blob/master/eipl/tutorials/rnn/libs/fullBPTT.py>[SOURCE] fullBPTT.py</a>" linenums="1" hl_lines="52"
class fullBPTTtrainer:
    def __init__(self,
                model,
                optimizer,
                device='cpu'):

        self.device = device
        self.optimizer = optimizer
        self.model = model.to(self.device)

    def save(self, epoch, loss, savename):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'train_loss': loss[0],
                    'test_loss': loss[1],
                    }, savename)

    def process_epoch(self, data, training=True):

        if not training:
            self.model.eval()

        total_loss = 0.0
        for n_batch, (x,y) in enumerate(data):
            x = x.to(self.device)
            y = y.to(self.device)

            state = None
            y_list = []
            T = x.shape[1]
            for t in range(T-1):
                y_hat, state = self.model(x[:,t], state)
                y_list.append(y_hat)

            y_hat = torch.permute(torch.stack(y_list), (1,0,2) )
            loss  = nn.MSELoss()(y_hat, y[:,1:] )
            total_loss += loss.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch+1)
```



<!-- #################################################################################################### -->
----
### Dataloader {#rnn_dataloader}
We describe DataLoader for learning image features and robot joint angles extracted by CAE with RNN.
As shown in lines 35 and 36, gaussian noise is added to the input data.
By training the model to minimize the error between the prediction values and the original data, the robot can predict appropriate motion commands even if noise is added in the real world.


```python title="<a href=https://github.com/ogata-lab/eipl/blob/master/eipl/tutorials/rnn/libs/dataloader.py>[SOURCE] dataloader.py</a>" linenums="1" hl_lines="20-21"
class TimeSeriesDataSet(Dataset):
    def __init__( self,
                  feats,
                  joints,
                  minmax=[0.1, 0.9],
                  stdev=0.02):

        self.stdev  = stdev
        self.feats  = torch.from_numpy(feats).float()
        self.joints = torch.from_numpy(joints).float()

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        y_feat  = self.feats[idx]
        y_joint = self.joints[idx]
        y_data  = torch.concat( (y_feat, y_joint), axis=-1)

        x_feat  = self.feats[idx]  + torch.normal(mean=0, std=self.stdev, size=y_feat.shape)
        x_joint = self.joints[idx] + torch.normal(mean=0, std=self.stdev, size=y_joint.shape)

        x_data = torch.concat( (x_feat, x_joint), axis=-1)

        return [x_data, y_data]
```




<!-- #################################################################################################### -->
----
### Training {#rnn_train}
The main program `train.py` is used to train RNN.
When the program is run, the trained weights (pth) and Tensorboard log files are saved in the `log` folder.
Please [see](https://github.com/ogata-lab/eipl/blob/master/eipl/tutorials/rnn/bin/train.py) the comments in the code for a detailed description of how the program works.


```bash 
$ cd eipl/tutorials/rnn/
$ python3 ./bin/train.py --device -1
[INFO] Set tag = 20230510_0134_03
================================
batch_size : 5
device : -1
epoch : 100000
log_dir : log/
lr : 0.001
model : LSTM
optimizer : adam
rec_dim : 50
stdev : 0.02
tag : 20230510_0134_03
vmax : 1.0
vmin : 0.0
================================
0%|               | 99/100000 [00:25<7:05:03,  3.92it/s, train_loss=0.00379, test_loss=0.00244
```




<!-- #################################################################################################### -->
----
### Inference {#rnn_inference}
Check that RNN has been properly trained using the test program `test.py`.
The arguments `filename` is the path of the trained weights file, `idx` is the index of the data you want to visualize,
To evaluate the generalization performance of the model, test data collected at [untrained location](../../teach/overview#task) are input and the true values are compared with the predicted values.
The figure below shows the `RNN` prediction results, where the left figure is the robot joint angles and the right figure is the image features.
The black dotted line in the figure represents the true value and the colored line represents the predicted value, and since they are almost identical, we can say that motion learning was done appropriately.


```bash 
$ cd eipl/tutorials/rnn/
$ python3 ./bin/test.py --filename ./log/20230510_0134_03/LSTM.pth --idx 4
$ ls output/
LSTM_20230510_0134_03_4.gif
```

![Predicted joint angles and image features using LSTM](img/cae-rnn/rnn_inference.gif){: .center}




<!-- #################################################################################################### -->
----
### Principal Component Analysis {#rnn_pca}
For an overview and concrete implementation of PCA, see [here](. /model/test.md#pca).


```bash
$ cd eipl/tutorials/rnn/
$ python3 ./bin/test_pca_rnn.py --filename log/20230510_0134_03/LSTM.pth
$ ls output/
PCA_LSTM_20230510_0134_03.gif
```

The figure figure shows the result of visualizing the internal state of RNN using PCA.
Each dotted line shows the time-series change of the RNN's internal state, and the internal state transitions sequentially starting from the black circle.
The transition trajectory of the internal state is called an attractor.
The color of each attractor is indicated by [object position](../teach/overview.md#task), with blue, orange, and green corresponding to taught positions A, C, and E, 
and red and purple corresponding to unlearned positions B and D.
Since the attractors are self-organized (aligned) according to the object position, it can be said that the behavior is learned (memorized) according to the object position.
In particular, since the attractors at the unlearned positions are generated between the taught positions, it is possible to generate unlearned interpolated motions by simply teaching and learning grasping motions with different object positions multiple times.


![Visualize the internal state of RNNs using Principal Component Analysis](img/cae-rnn/rnn_pca.webp){: .center}
