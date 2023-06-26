# FAQ

## Q. Motion is not smoothly.

1. Mixing predictive data
    In order to generate stable/smooth motion against real-world noise, the sensor information at time $t$ is mixed with the predicted value of RNN at the previous time $t-1$ in a specific ratio and then input to RNN.
    This process is equivalent to a low-pass filter, and even if the robot's sensor values are noisy, the predicted values from the previous time can be used as a supplement to predict stable motion commands.
    Note that if the mixing factor (`input_param`) is too small, it becomes difficult to modify the motion based on real-world sensor information, and the robustness against position changes decreases.
    If `input_param=0.0`, the motion will be generated using only the sensor information acquired at the initial time.
    The following is an example implementation, in which data is mixed with the robot's camera image and joint angles.

    ```python
    x_image, x_joint = robot.get_sensor_data()

    if loop_ct > 1:
        x_image = x_image * input_param + y_image * (1.0-input_param)
        x_joint = x_joint * input_param + y_joint * (1.0-input_param)

    y_image , y_joint, state = mode(x_image, x_joint, state)
    ```


## Q. Predicted image is abnormal

1. Fix camera parameters
    When the trained model is applied to a real robot, the problem occurs that no objects are seen in the predicted image or the predicted image is noisy.
    This may be due to the fact that camera parameters (e.g. white balance) are automatically changed, fix the camera parameters at the time of `motion teaching`.
    Or adjust the `inference` camera parameters to get the same visual image as the `motion teaching`.

   

## Q. Not focusing attention on the object.

1. Adjusting the camera position

    It is recommended that the robot's body (hand or arm) and the object be constantly displayed in the image in order to acquire stable attention and motion.
    When attention is directed to the robot's body and the object part, it is easier to learn the time-series relationship between the both.
    
2. Enlarge the object

    Therefore, either make the object physically larger, crop the image around the object, or move the camera closer to the object.

3. Re-training the mode
    The initial weights of the model may cause the objects to be inattentive.
    Training multiple times with the same parameters yields good results 3 out of 5 times.


## Q. Customize the data loader

It is possible to add/remove any sensor by changing the number of data to pass the `MultimodalDataset` class or by changing some of the input/output definitions.
The following shows an example of adding a new torque sensor.

```python
class MultimodalDataset(Dataset):
    def __init__(self, images, joints, torque, stdev=0.02):
        pass

    def __getitem__(self, idx):
        return [[x_img, x_joint, x_torque], [y_img, y_joint, y_torque]]
```


