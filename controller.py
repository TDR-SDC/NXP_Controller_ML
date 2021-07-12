# !/usr/bin/env python3

from math import atan
import time
import cv2
import numpy as np
import os

from numpy.lib import stride_tricks
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32

import tensorflow as tf


input_shape = [216, 216, 3]


class Controller(Node):
    def __init__(self):
        super().__init__('Controller')
        self.bridge = CvBridge()

        # Subscriptions
        self.subscription = self.create_subscription(
            Image,
            '/trackImage0/image_raw',
            self.listener_callback,
            qos_profile_sensor_data)
        
        self.sign_subscription = self.create_subscription(
            Int32,
            '/traffic_sign',
            self.trafSign_callback,
            1)

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cupcar0/cmd_vel', 10)

        # self.model = tf.keras.models.load_model('/home/klrshak/ros2ws/src/aim_line_follow/aim_line_follow/NXP_Controller_ML/weights/Weights_6-old')
        self.model = tf.keras.models.load_model('/home/klrshak/ros2ws/src/aim_line_follow/aim_line_follow/NXP_Controller_ML/weights/Weights_5')
        print("*************WEIGHTS LOADED****************")
        
        # Declarations
        self.speed_vector = Vector3()
        self.steer_vector = Vector3()
        self.cmd_vel = Twist()
        self.trafSign_ID = -1
        self.Left_turn_interval = 0
        self.Right_turn_interval = 0

    def listener_callback(self, data):
        img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (216, 216))
        img = np.reshape(img, (-1, 216, 216, 3))
        # cv2.imwrite('image.jpg', img)
        outputs = self.model.predict(img)
        # outputs = outputs
        print(outputs)

        # velocity = outputs[0]
        # velocity *= 1.5
        # velocity = max(0, velocity)

        steer = - atan(outputs)
        velocity = 0.5
        
        steer, velocity = self.state_traffic_sign(steer, velocity)

        self.speed_vector.x = float(velocity)# vel
        self.steer_vector.z = float(steer) #steer
        
        self.cmd_vel.linear = self.speed_vector
        self.cmd_vel.angular = self.steer_vector
        self.cmd_vel_publisher.publish(self.cmd_vel)

    def trafSign_callback(self, data):
        self.trafSign_ID = data  #check data structure

    def state_traffic_sign(self, steer, velocity):
        if self.trafSign_ID == 0 or self.turn_interval > 0: #left Turn
            steer = 0.2
            if self.Left_turn_interval == 0:
                self.Left_turn_interval = time.time()
        elif self.trafSign_ID == 1 or self.Right_turn_interval > 0:
            steer = -0.2
            if self.Right_turn_interval == 0: 
                self.Right_turn_interval = time.time()
        elif self.trafSign_ID == 2:
            velocity = 0
        
        if (time.time() - self.Left_turn_interval) > 0.3:
            self.Left_turn_interval = 0
        if (time.time() - self.Right_turn_interval) > 0.3:
            self.Right_turn_interval = 0

        return steer, velocity

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()