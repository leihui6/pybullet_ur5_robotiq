import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R

from utilities import Models, Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class ClutteredPushGrasp:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, models: Models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.setGravity(0, 0, -10)
        p.setGravity(0, 0, 0)
        # self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        # self.boxID = p.loadURDF("./urdf/skew-box-button.urdf",  [0.0, 0.0, 0.0], p.getQuaternionFromEuler([0, 1.5706453, 0]))
                                # p.getQuaternionFromEuler([0, 0, 0])
        #                         useFixedBase=True,
        #                         flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

        # self.cameraID = p.loadURDF('./urdf/3DCamera.urdf', [0,0,0], p.getQuaternionFromEuler([0, 0, 0]))
        self.cameraID = p.loadURDF('./urdf/small_sphere.urdf', [0,0,0], p.getQuaternionFromEuler([0, 0, 0]))
        self.modelID = self.add_3D_model(f'./meshes/scanObj/stanford-bunny.stl')

        # For calculating the reward
        self.box_opened = False
        self.btn_pressed = False
        self.box_closed = False

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def compute_rotation_matrix_to_center(self, cameraPos, theta, beta):
        theta, beta = np.deg2rad(theta), np.deg2rad(beta)

        to_camera_pos_vec = self.normalize(np.array(cameraPos))
        rot_axis_1 = np.cross(np.array([0,0,1]), to_camera_pos_vec)
        rot_axis_1 = self.normalize(rot_axis_1)

        r1 = R.from_rotvec((theta + np.pi) * rot_axis_1)
        quaternion = r1.as_quat()
        return quaternion


    def add_3D_model(self, filename:str):
        shift = [0, -0.02, 0]
        meshScale = [0.1, 0.1, 0.1]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=filename,
                                    rgbaColor=[1, 1, 1, 1],
                                    specularColor=[0.4, .4, 0],
                                    visualFramePosition=shift,
                                    meshScale=meshScale)
        
        modelID = p.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                    #   baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=[0,0,0],
                      useMaximalCoordinates=True)
        return modelID


    def update_camera(self, R, theta, beta):
        r_theta, r_beta = np.deg2rad(theta), np.deg2rad(beta)
        x = R * np.sin(r_theta) * np.cos(r_beta)
        y = R * np.sin(r_theta) * np.sin(r_beta)
        z = R * np.cos(r_theta)
        cameraPos = [x,y,z]
        # p.addUserDebugLine(cameraPos, [0,0,0], [1, 0, 0], 1)
        
        quaM = self.compute_rotation_matrix_to_center(cameraPos, theta, beta)
        p.resetBasePositionAndOrientation(self.cameraID, cameraPos, quaM)

        targetPos = [0,0,0]
        cameraupPos = [0,0,1.0] 
        view_matrix = self.camera.update_pose(cameraPos, targetPos, cameraupPos)
        return view_matrix

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(0.1)


    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length


    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control

        @return: the image of the RGB or depth
        """
        assert control_method in ('joint', 'end')
        # self.robot.move_ee(action[:-1], control_method)
        # self.robot.move_gripper(action[-1])
        # for _ in range(120):  # Wait for a few steps
        self.step_simulation()

        # reward = self.update_reward()
        # done = True if reward == 1 else False
        # info = dict(box_opened=self.box_opened, btn_pressed=self.btn_pressed, box_closed=self.box_closed)
        # return self.get_observation(), reward, done, info
        
        return self.get_observation() # rgbImage, depthImage, seg


    def update_reward(self):
        reward = 0
        if not self.box_opened:
            if p.getJointState(self.boxID, 1)[0] > 1.9:
                self.box_opened = True
                print('Box opened!')
        elif not self.btn_pressed:
            if p.getJointState(self.boxID, 0)[0] < - 0.02:
                self.btn_pressed = True
                print('Btn pressed!')
        else:
            if p.getJointState(self.boxID, 1)[0] < 0.1:
                print('Box closed!')
                self.box_closed = True
                reward = 1
        return reward


    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, pointcloud, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, pc = pointcloud, seg=seg))
        else:
            assert self.camera is None
        # obs.update(self.robot.get_joint_obs())
        # print (obs)
        return obs


    def reset_box(self):
        # p.setJointMotorControl2(self.boxID, 0, p.POSITION_CONTROL, force=1)
        # p.setJointMotorControl2(self.boxID, 1, p.VELOCITY_CONTROL, force=0)
        pass


    def reset(self):
        self.robot.reset()
        self.reset_box()
        return self.get_observation()


    def close(self):
        p.disconnect(self.physicsClient)
