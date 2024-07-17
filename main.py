import os

import numpy as np
import pybullet as p
import cv2

from tqdm import tqdm
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera
import time
import math
import open3d as o3d

import matplotlib.pyplot as plt

def user_control_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq85((1.0, 0.0, 0), (0.0, 0.0, 0.0))
    camera = Camera(cam_pos = (1, 1, 1), cam_tar= (0, 0, 0), cam_up_vector = (0, 0, 1),
                    near = 0.1, far = 5, 
                    size = (320, 320), fov = 40)
    # camera = Camera(robot_type=robot,
    #                 near = 0.1, far = 5, 
    #                 size = (320, 320), fov = 40)
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)
    env.reset()
    env.SIMULATION_STEP_DELAY = 0

    # the working distance of the camera
    R = 0.8
    # while True:
        # obs = env.step(env.read_debug_parameter(), 'end')
        # pass

    for theta in range(10,90,10): # height
        for beta in range(0,360,10): # XY-plane
            env.update_camera(R, theta, beta)
            env.robot.reset()
            obs = env.step(env.read_debug_parameter(), 'end')
            # input()
            # pointcloud = obs['pc'] 
            # o3d.visualization.draw_geometries([pointcloud])

        # rgbImage = obs['rgb']
        # depthImage = obs['depth'] 
        # pointcloud = obs['pc'] 
        # o3d.visualization.draw_geometries([pointcloud])

        # cv2.imshow('color-image',  rgbImage)
        # plt.imshow(depthImage)
        # plt.show()
        # exit()
        # obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
        # print(obs, reward, done, info)


if __name__ == '__main__':
    user_control_demo()
