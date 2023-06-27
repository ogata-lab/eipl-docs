# FAQ

## Q. Motion is not smooth.

To generate stable and smooth motion against real-world noise, the sensor information at time $t$ is mixed with the predicted value from the previous time $t-1$ in a certain ratio and then fed into the RNN. This process acts as a low-pass filter, allowing the predicted values from the previous time to supplement the motion prediction even in the presence of noisy sensor values. It is important to note that if the mixing factor (`input_param`) is too small, it becomes difficult to modify the motion based on real sensor information, resulting in reduced robustness to position changes. When `input_param=0.0`, the motion is generated based solely on the sensor information obtained at the initial time. Below is an example implementation that shows data mixing using the robot's camera image and joint angles.

```python
x_image, x_joint = robot.get_sensor_data()

if loop_ct > 1:
    x_image = x_image * input_param + y_image * (1.0-input_param)
    x_joint = x_joint * input_param + y_joint * (1.0-input_param)

y_image , y_joint, state = mode(x_image, x_joint, state)
```


## Q. The predicted image looks abnormal.

When applying a trained model to a real robot, there may be problems such as objects not being visible in the predicted image, or the image being noisy. This could be due to automatic changes in camera parameters (e.g. white balance). To avoid this, it is recommended to fix the camera parameters during motion teaching, or to adjust the inference camera parameters to match the visual image during motion teaching.

   

## Q. The model does not focus on the object.

1. Adjust the camera position

    It is recommended to make sure that the robot's body (hand or arm) and the object are consistently displayed in the image to facilitate stable attention and movement. When attention is directed to both the robot's body and the object, it becomes easier to learn the temporal relationship between them.
    
2. Enlarging the object

    Consider physically enlarging the object, cropping the image around the object, or moving the camera closer to the object.

3. Retrain the model

    The initial weights of the model may cause inattention to the objects. Training the model multiple times with the same parameters has yielded positive results 3 out of 5 times.



## Q. How can I customize the data loader?

It is possible to add or remove any sensor by adjusting the amount of data passed to the `MultimodalDataset` class or by modifying the input/output definitions. Here is an example that shows how to add a new torque sensor.

```python
class MultimodalDataset(Dataset):
    def __init__(self, images, joints, torque, stdev=0.02):
        pass

    def __getitem__(self, idx):
        return [[x_img, x_joint, x_torque], [y_img, y_joint, y_torque]]
```


