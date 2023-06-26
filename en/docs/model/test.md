# Inference {#test}
<!-- #################################################################################################### -->
## Off-line inference 
Check that SARNN has been properly trained using the test program `test.py`.
The argument `filename` is the path of the trained weights file and `idx` is the index of the data to be visualized.
The `input_param` is a mixing coefficient to generate stable behavior against real-world noise. The sensor information at time $t$ is mixed with the predictions of the model at the previous time $t-1$ in a certain ratio and input to the model.
This process is equivalent to a low-pass filter, and even if the robot's sensor values are noisy, the predicted values from the previous time can be used as a supplement to predict stable motion commands.
Note that if the mixing coefficient is too small, it becomes difficult to modify the motion based on real-world sensor information, and the robustness against position changes decreases.


```bash
$ cd eipl/tutorials/sarnn/
$ python3 bin/test.py --filename ./log/20230521_1247_41/SARNN.pth --idx 4 --input_param 1.0

images shape:(187, 128, 128, 3), min=0, max=255
joints shape:(187, 8), min=-0.8595600128173828, max=1.8292399644851685
loop_ct:0, joint:[ 0.00226304 -0.7357931  -0.28175825  1.2895856   0.7252841   0.14539993
-0.0266939   0.00422328]
loop_ct:1, joint:[ 0.00307412 -0.73363686 -0.2815826   1.2874944   0.72176594  0.1542334
-0.02719587  0.00325996]
.
.
.

$ ls output/
SARNN_20230521_1247_41_4_1.0.gif
```

The following figure shows the inference results at the unlearned position ([point D](../teach/overview.md#task)). From left to right, the input image, the predicted image, and the predicted joint angles (dotted lines are true values). The blue points in the input image are the points of interest extracted from the image, and the red points are the points of interest predicted by the RNN, indicating that the joint angle is predicted while focusing on the robot hand and the grasped object.

![results_of_SARNN](img/sarnn-rt_4.webp){: .center}



<!-- #################################################################################################### -->
----
## Principal Component Analysis {#pca}
In deep predictive learning, it is recommended to visualize the internal representation using Principal Component Analysis (PCA)[@hotelling1933analysis] in order to preliminarily examine whether the trained model has generalization performance.
In order to acquire generalization motion with small data, it is necessary to embed the motion in the RNN's internal state, and the internal state should be self-organized (structured) for each teaching motion.
Hereafter, we use PCA to compress the internal state of the RNN to a lower dimension, and visualize the elements (first through third principal components) that represent the characteristics of the data to verify how the sensorimotor information (images and joint angles) are represented.


The following program is a partial excerpt of the inference and PCA process.
First, input test data into the model and store the internal state `state` of the RNN at each time as a list.
In the case of [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html), `hidden state` and `cell state` are returned as `state`.
Here we use `state` for visualization and analysis.
Next, we transform the shape of `state`, from [number of data, time series length, number of dimensions of state] to [number of data x time series length, number of dimensions of state] in order to compare the internal state at each object position.
Finally, the high-dimensional `state` is compressed into low-dimensional information (3 dimensions) by applying PCA as shown in line 12.
By restoring the compressed principal component `pca_val` to its original shape [number of data, time series length, 3 dim], we can visualize the relationship between object position and internal state by coloring each object position and plotting it in 3D space.


```python title="<a href=https://github.com/ogata-lab/eipl/blob/master/eipl/tutorials/sarnn/bin/test_pca_sarnn.py>[SOURCE] test_pca_rnn.py</a>" linenums="1" hl_lines="12"
states = tensor2numpy( states )
# Reshape the state from [N,T,D] to [-1,D] for PCA of RNN.
# N is the number of datasets
# T is the sequence length
# D is the dimension of the hidden state
N,T,D  = states.shape
states = states.reshape(-1,D)

# plot pca
loop_ct = float(360)/T
pca_dim = 3
pca     = PCA(n_components=pca_dim).fit(states)
pca_val = pca.transform(states)
# Reshape the states from [-1, pca_dim] to [N,T,pca_dim] to
# visualize each state as a 3D scatter.
pca_val = pca_val.reshape( N, T, pca_dim )

fig = plt.figure(dpi=60)
ax = fig.add_subplot(projection='3d')

def anim_update(i):
    ax.cla()
    angle = int(loop_ct * i)
    ax.view_init(30, angle)

    c_list = ['C0','C1','C2','C3','C4']
```



Use `test_pca_sarnn.py` for the program to visualize the internal state using PCA.
The argument filename is the path of the weights file.

```bash
$ cd eipl/tutorials/sarnn/
$ python3 ./bin/test_pca_sarnn.py --filename log/20230521_1247_41/SARNN.pth
$ ls output/
PCA_SARNN_20230521_1247_41.gif
```

The figure below shows the inference result of SARNN. Each dotted line shows the time series change of the internal state.
The color of each attractor corresponds to [object position](../teach/overview.md#task), with blue, orange, and green corresponding to taught positions A, C,and E,
and red and purple to untrained positions B and D.
Since the attractors are self-organized (aligned) according to the object position, it can be said that the behavior is learned (memorized) according to the object position.
In particular, since the attractors at unlearned positions are generated between taught positions, it is possible to generate unlearned interpolated motions by simply teaching and learning grasping motions with different object positions multiple times.

![Visualize_the_internal_state_of_SARNN_using_Principal_Component_Analysis](img/sarnn_pca.webp){: .center}



<!-- #################################################################################################### -->
----
## Online Motion Generation {#online}
The following describes an online motion generation method using a real robot with pseudo code.
The robot can generate sequential motions based on sensor information by repeating steps 2-5 at a specified sampling rate.


1. **Model loading (line 21)**

    After defining the model, load the trained weights.

2. **Get and normalize sensor information (line 36)**

    Get the robot sensor information and perform the normalization process.
    For example, if you are using ROS, the Subscribed image and joint angles as `raw_image` and `raw_joint`.
    

3. **Inference (line 49)**

    Predict the image `y_image` and joint angle `y_joint` at the next time using the normalized image `x_img` and joint angle `x_joint`.
    

4. **Send command (line 59)**

    By using the predicted joint angle `pred_joint` as the robot's motor command, the robot can generate sequential motions.
    In the case of ROS, by publishing the joint angles to the motors, the robot controls each motor based on the motor command.

5. **Sleep (line 63)**

    Finally, timing is adjusted by inserting a sleep process to perform inference at the specified sampling rate. The sampling rate should be the same as during training data collection.



```python title="online.py" linenums="1" hl_lines="21-24 36-42 49-50 59-61 63-65"
parser = argparse.ArgumentParser()
parser.add_argument('--model_pth', type=str, default=None)
parser.add_argument('--input_param',type=float, default=0.8 )
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.model_pth)[0]
params = restore_args( os.path.join(dir_name, 'args.json') )

# load dataset
minmax = [params['vmin'], params['vmax']]
joint_bounds = np.load( os.path.join( os.path.expanduser('~'), '.eipl/grasp_bottle/joint_bounds.npy') )

# define model
model = SARNN( rec_dim=params['rec_dim'],
               joint_dim=8,
               k_dim=params['k_dim'],
               heatmap_size=params['heatmap_size'],
               temperature=params['temperature'])

# load weight
ckpt = torch.load(args.model_pth, map_location=torch.device('cpu') )
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Inference
# Set the inference frequency; for a 10-Hz in ROS system, set as follows.
freq = 10 # 10Hz
rate = rospy.Rate(freq)
image_list, joint_list = [], []
state = None
nloop = 200 # freq * 20 sec
for loop_ct in range(nloop):
    start_time = time.time()

    # load data and normalization
    raw_images, raw_joint = robot.get_sensor_data()
    x_img = raw_images[loop_ct].transpose(2,0,1)
    x_img = torch.Tensor( np.expand_dims(x_img, 0 ) )
    x_img = normalization( x_img, (0,255), minmax )
    x_joint = torch.Tensor( np.expand_dims(raw_joint, 0 ) )
    x_joint = normalization( x_joint, joint_bounds, minmax )

    # closed loop
    if loop_ct > 0:
        x_img   = args.input_param*x_img   + (1.0-args.input_param)*y_image
        x_joint = args.input_param*x_joint + (1.0-args.input_param)*y_joint

    # predict rnn
    y_image, y_joint, state = rnn_model(x_img, x_joint, state)

    # denormalization
    pred_image = tensor2numpy(y_image[0])
    pred_image = deprocess_img(pred_image, cae_params['vmin'], cae_params['vmax'])
    pred_image = pred_image.transpose(1,2,0)
    pred_joint = tensor2numpy(y_joint[0])
    pred_joint = normalization(pred_joint, minmax, joint_bounds)

    # send pred_joint to robot
    # send_command(pred_joint)
    pub.publish(pred_joint)

    # Sleep to infer at set frequency.
    # ROS system
    rate.sleep()
```
