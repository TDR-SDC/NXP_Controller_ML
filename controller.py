# !/usr/bin/env python3

import time
import cv2
import numpy as np
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from math import atan2
# import tensorflow as tf
# from tensorflow import keras


input_shape = [216, 216, 3]


class Controller(Node):
    def __init__(self):
        super().__init__('Controller')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/trackImage0/image_raw',
            self.listener_callback,
            qos_profile_sensor_data)
        
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cupcar0/cmd_vel', 10)

        # self.model = keras.models.load_model('/home/anikets2002/ros2ws/src/aim_line_follow/aim_line_follow/NXP_Controller/Weights/Weights_80')


    def listener_callback(self, data):
        img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (216, 216))
        img = np.reshape(img, (-1, 216, 216, 3))
        cv2.imwrite('image.jpg', img)

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()