# !/usr/bin/env python3

import time
import cv2
import numpy as np
import os
import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from math import atan


# todo Initialising training location
training_dir = r'/data'


# todo Initializing PyGame and Joystick
pygame.init()
pygame.joystick.init()
_joystick = pygame.joystick.Joystick(0)
_joystick.init()

def get_user_input():
    pygame.event.get()
    user_input = _joystick.get_axis(3)

    return user_input

class DataLoader(Node):
    def __init__(self):
        super().__init__("DataLoader")
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/trackImage0/image_raw',
            self.listener_callback,
            qos_profile_sensor_data)
        
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cupcar0/cmd_vel', 10)
        self.list_imgs = []
        self.train_data = []
        self.file_count = 0

        self.speed_vector = Vector3()
        self.steer_vector = Vector3()
        self.cmd_vel = Twist()
        self.Pause = False


    def listener_callback(self, data):
        usr_in = get_user_input()

        steer = - atan(usr_in)

        if not self.Pause:  # self.Pause Mechanism
            path = r'/home/klrshak/ros2ws/src/aim_line_follow/aim_line_follow/NXP_Controller_ML/data/'
            file_name = r'training_data_{}.npy'.format(self.file_count)
            img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (216, 216))
            # img = np.array(img)
            self.train_data.append([img, usr_in])
            # print(len(self.train_data))

            if len(self.train_data)%3000==0:
                np.save(path + file_name, self.train_data)
                print("Saved {}".format(file_name))
                self.file_count +=1
                self.train_data = []

            self.speed_vector.x = float(0.5)
            self.steer_vector.z = float(steer)

            self.cmd_vel.linear = self.speed_vector
            self.cmd_vel.angular = self.steer_vector
            self.cmd_vel_publisher.publish(self.cmd_vel)

            cv2.imshow("image", img)
            cv2.waitKey(1)
            pygame.event.get()
            if _joystick.get_button(5):  # 5 is right bumper
                self.Pause = True
                print('---------------PAUSED!!!--------------')
                time.sleep(0.5)


        pygame.event.get()
        if _joystick.get_button(5):  # 5 is right Trigger
            self.Pause = False
            print('Starting Up.............')
            time.sleep(0.5)


def main(args=None):
    rclpy.init(args=args)
    dataloader = DataLoader()
    rclpy.spin(dataloader)

    dataloader.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
