import pybullet as p
import glob
from collections import namedtuple
from attrdict import AttrDict
import functools
# import torch
import cv2
from scipy import ndimage
import numpy as np
import PIL.Image as Image

import open3d as o3d

class Models:
    def load_objects(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        return NotImplementedError


class YCBModels(Models):
    def __init__(self, root, selected_names: tuple = ()):
        self.obj_files = glob.glob(root)
        self.selected_names = selected_names
        self.visual_shapes = []
        self.collision_shapes = []

    def load_objects(self):
        shift = [0, 0, 0]
        mesh_scale = [1, 1, 1]

        for filename in self.obj_files:
            # Check selected_names
            if self.selected_names:
                in_selected = False
                for name in self.selected_names:
                    if name in filename:
                        in_selected = True
                if not in_selected:
                    continue
            print('Loading %s' % filename)
            self.collision_shapes.append(
                p.createCollisionShape(shapeType=p.GEOM_MESH,
                                       fileName=filename,
                                       collisionFramePosition=shift,
                                       meshScale=mesh_scale))
            self.visual_shapes.append(
                p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=filename,
                                    visualFramePosition=shift,
                                    meshScale=mesh_scale))

    def __len__(self):
        return len(self.collision_shapes)

    def __getitem__(self, idx):
        return self.visual_shapes[idx], self.collision_shapes[idx]


class Camera:
    def __init__(self, cam_pos, cam_tar, cam_up_vector, near, far, size, fov):
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        self.aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, cam_up_vector)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)


    def update_pose(self, cameraPos, targetPos, cameraupPos):
        self.aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix(
                cameraEyePosition = cameraPos,
                cameraTargetPosition = targetPos,
                cameraUpVector = cameraupPos)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)
        return _view_matrix


    def rgbd_2_world(self, w, h, d):
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1))
        position = self.tran_pix_world @ pix_pos
        position /= position[3]
        return position[:3]

    def rgb_2_Image(self, rgb):
        rgbImage = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgbImage

    def depth_2_Image(self, depth):
        depth_buffer_tiny = np.reshape(depth, [self.width, self.height])
        depImg = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer_tiny)
        depImg = np.asanyarray(depImg).astype(np.float32) * 1000.  # mm -> meter
        
        depImg = (depImg.astype(np.uint16)) # save as png
        # depImg = Image.fromarray(depImg)
        # depImg.save('./test.png')

        # depImg = (depImg.astype(np.uint8)) # save as jpg
        # depImg = Image.fromarray(depImg)
        # depImg.save('./test.jpg')
        return depImg


    def rgb_depth_2_Pointcloud(self, rgbImage, depthImage):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgbImage), 
                                                                        o3d.geometry.Image(depthImage),
                                                                        convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
        # print (self.projection_matrix)
        # print (type(self.projection_matrix))
        P_0, P_1 = self.projection_matrix[0], self.projection_matrix[5]
        intrinsic.set_intrinsics(width=self.width, height=self.height,
                                 fx = P_0 * self.width / 2, fy = P_1 * self.height / 2,
                                 cx = self.width / 2, cy = self.height / 2)
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        # o3d.visualization.draw_geometries([point_cloud])
        return point_cloud
        

    def shot(self):
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   )
        _rgbImage = self.rgb_2_Image(rgb)
        _depthImage = self.depth_2_Image(depth)
        _pointcloud = self.rgb_depth_2_Pointcloud(_rgbImage, _depthImage)
        # return rgb, depth, seg
        return _rgbImage, _depthImage, _pointcloud, seg


    def rgbd_2_world_batch(self, depth):
        # reference: https://stackoverflow.com/a/62247245
        x = (2 * np.arange(0, self.width) - self.width) / self.width
        x = np.repeat(x[None, :], self.height, axis=0)
        y = -(2 * np.arange(0, self.height) - self.height) / self.height
        y = np.repeat(y[:, None], self.width, axis=1)
        z = 2 * depth - 1

        pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
        position = self.tran_pix_world @ pix_pos.T
        position = position.T
        # print(position)

        position[:, :] /= position[:, 3:4]

        return position[:, :3].reshape(*x.shape, -1)

def pybullet_2_world(pc, view_matrix):
    #left hand -> right hand
    depth=np.asarray(pc.points)
    for i in range(len(depth)):
        depth[i][0]=- depth[i][0]
    pc.points = o3d.utility.Vector3dVector(depth)
    R1 = view_matrix[:3, :3].T
    T = view_matrix[:3, 3]
    pc = pc.translate((T[0], T[1], T[2]), relative=True)
    pc = pc.rotate(R1, center=(0, 0, 0))
    return pc