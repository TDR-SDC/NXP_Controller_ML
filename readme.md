# NXP Car Controller end-to-end pipeline using imitation learning
Developed a Deep-learning based framework for visual navigation using Imitation Learning in an urban environment laid
with obstacles such as pedestrians, barricades and other cars. Deployed End-to-End CNN model with regularization
technique with ROS2 in a simulated environment.

### For Object Detection
- Significantly improved accuracy from a very small dataset of 1000 images for 5 categories, by creating custom dataset
with image augmentation and data scraping; And smart pre-training of YOLOv4 Tiny.
Creating NXP car control using convolutional neural networks

## Data
- Data Files: https://drive.google.com/drive/folders/1TLdkgaS9dncPFjXhM6tqCrHvLh1JJjY7?usp=sharing

## Training Procedure
- First run the environment simulator
- Then run the data generation file to capture data from the simulator
- Control the simulator with the joystick to create data
```shell
python3   train_data_gen.py
```

- After enough data is generated we can move on to train our model using:
  
```shell
python3   train.py
```

## Deployment 
- After trainning the learned controller can be deployed by using
```shell
python3   controller.py
```
