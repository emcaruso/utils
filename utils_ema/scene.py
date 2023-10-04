import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import os
import time
import sys
import itertools
import random
import argparse
import copy
import pathlib
import wandb


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

current = os.path.dirname(os.path.realpath(__file__))

# utils_path = current+"/src/utils"
# sys.path.insert(1, utils_path)

from utils_ema.camera_cv import *
from utils_ema.geometry_pose import *
from utils_ema.geometry_euler import *
from utils_ema.figures import *
from utils_ema.plot import *
from utils_ema.mesh import Mesh, read_mesh, AABB

try:
    from .camera_cv import *
    from .geometry_pose import *
    from .geometry_euler import *
    from .figures import *
    from .plot import *
    from .mesh import Mesh, read_mesh, AABB
except:
    from camera_cv import *
    from geometry_pose import *
    from geometry_euler import *
    from figures import *
    from plot import *
    from mesh import Mesh, read_mesh, AABB


class Scene():
    def __init__(self, data_dir, data_dataset, list_images, frames=None, device='cpu', load_images=None ):
        self.device=device
        self.data_dir = data_dir+"/"+data_dataset
        if not os.path.isdir(self.data_dir): raise TypeError("the path: "+self.data_dir+" doesn't exists")
        if frames is None: self.mode = 'static'
        else: self.mode = 'video'
        self.mesh_dir = self.data_dir+"/meshes"
        self.name_dataset = data_dataset
        self.list_images = list_images
        self.frames = frames
        print("dataset path: ",self.data_dir)
        if os.path.isfile(self.data_dir+"/"+self.mode+"/data.npz"):
            self.data_type = "Blender"
            self.init_blender(load_images)
        elif os.path.isfile(self.data_dir+"/"+self.mode+"/data_basler.npz"):
            self.data_type = "Basler"
            self.init_basler(load_images)
        else:
            print("not valid scene")

    def init_basler(self, load_images):
        print("Loading Cameras from Basler data: "+self.name_dataset)
        self.npz_path = self.data_dir+"/"+self.mode+"/data_basler.npz"
        if not os.path.isfile(self.npz_path): raise TypeError("the file: "+self.npz_path+" doesn't exists")
        self.npz = np.load(self.npz_path, allow_pickle=True)
        self.camera_names = list(self.npz.keys())
        self.n_objects = 0
        self.objects = []
        # frames
        if self.mode == 'static':
            pass
        elif self.frames == "All":
            frames_available = [ int(f.split("_")[-1]) for f in list(os.listdir(self.data_dir+"/video")) if os.path.isdir(self.data_dir+"/video/"+f) ]
            self.frames = frames_available
            self.n_frames = len(self.frames)
        elif len(self.frames) == 1:
            self.frames = [self.frames[0]]
        else:
            self.frames = range(self.frames[0],self.frames[1])
            frames_available = [ int(f.split("_")[-1]) for f in list(os.listdir(self.data_dir+"/video")) if os.path.isdir(self.data_dir+"/video/"+f) ]
            for frame in self.frames: assert(frame in frames_available)
            self.n_frames = len(self.frames)
        self.n_cameras = len(self.camera_names)
        self.cams = [[None]*self.n_cameras for _ in range(self.n_frames)]
        for j, cam_arr in enumerate(list(self.npz.values())):
            for i, frame in enumerate(self.frames):
                cam = cam_arr.tolist()
                cam_name = "Cam_"+str(j).zfill(3)
                image_paths = self.get_imagepaths_from_path(self.data_dir+"/video/frame_"+str(frame).zfill(3), cam_name)
                new_cam = cam.clone(same_intr=True, same_pose=True, image_paths=image_paths, name=cam_name)
                if load_images is not None: new_cam.load_images(load_images, self.device)
                self.cams[i][j] = new_cam
                

    def init_blender(self, load_images):
        print("Loading Cameras from Blender data: "+self.name_dataset)
        self.npz_path = self.data_dir+"/"+self.mode+"/data.npz"
        if not os.path.isfile(self.npz_path): raise TypeError("the file: "+self.npz_path+" doesn't exists")
        self.npz = np.load(self.npz_path, allow_pickle=True)
        self.camera_names = list(self.npz["cameras"].tolist().keys())
        self.camera_attributes = list(self.npz["cameras"].tolist().values())
        self.n_cameras = len(self.camera_names)
        self.tracked_objects = self.npz["objects"].tolist()
        self.n_objects = len(self.tracked_objects)
        self.cams = [[None]*self.n_cameras]
        self.objects = [[None]*self.n_objects]
        # frames
        if self.mode == 'static':
            self.frames = [self.npz["frame_static"].tolist()]
        elif len(self.frames) == 1:
            self.frames = [self.frames[0]]
        else:
            self.frames = range(self.frames[0],self.frames[1])
            frames_available = [ int(f.split("_")[-1]) for f in list(os.listdir(self.data_dir+"/video")) if os.path.isdir(self.data_dir+"/video/"+f) ]
            for frame in self.frames: assert(frame in frames_available)
            self.n_frames = len(self.frames)


        # get frames
        assert( os.path.isdir(self.data_dir+"/video") )
        self.cams = [[None]*self.n_cameras for _ in range(self.n_frames)]
        self.objects = [[None]*self.n_objects for _ in range(self.n_frames)]

        # load cameras
        for j, (cam_name,cam_dict) in enumerate(self.npz["cameras"].tolist().items()) :
            for i, frame in enumerate(self.frames):
                if isinstance(cam_dict["pose"], Pose): pose = cam_dict["pose"]
                else: pose = cam_dict["pose"][frame]
                image_paths = self.get_imagepaths_from_path(self.data_dir+"/video/frame_"+str(frame), cam_name)
                cam = Camera_cv(intrinsics= cam_dict["intrinsics"], pose=pose, frame=frame, image_paths=image_paths, name=cam_name+"_"+str(frame), device=self.device) 
                if load_images is not None: cam.load_images(load_images, self.device)
                self.cams[i][j] = cam
        self.cams_flatten = [ cam for frame in self.cams for cam in frame]

        # load objects
        meshlist = [ file for file in os.listdir(self.mesh_dir) if pathlib.Path(file).suffix == ".obj"]
        for j, (obj_name, obj_dict) in enumerate(self.tracked_objects.items()):
            obj_file = obj_name+".obj"
            assert(obj_file in meshlist)
            mesh = read_mesh(self.mesh_dir+"/"+obj_file, device=self.device)
            mesh.compute_connectivity()
            for i, frame in enumerate(self.frames):
                if isinstance(obj_dict["pose"], Pose): pose = obj_dict["pose"]
                else: pose = obj_dict["pose"][frame]
                obj = { "mesh": mesh, "pose": pose.to(self.device) }
                self.objects[i][j] = obj
        self.objects_flatten = [ obj for frame in self.objects for obj in frame]

        # check units
        units = self.cams[0][0].pose.units
        for frame in self.cams:
            for cam in frame:
                assert(cam.pose.units==units)
        for frame in self.objects:
            for obj in frame:
                assert(obj["pose"].units==units)


        # elif self.mode == 'static':

        #     # load cameras
        #     for j, (cam_name,cam_dict) in enumerate(self.npz["cameras"].tolist().items()) :
        #         image_paths = self.get_imagepaths_from_path(self.data_dir+"/static", cam_name)
        #         cam = Camera_cv(intrinsics= cam_dict["intrinsics"], pose=cam_dict["pose"], image_paths=image_paths, name=cam_name) 
        #         self.cams[0][j] = cam

        #     # load objects
        #     meshlist = [ file for file in os.listdir(self.mesh_dir) if pathlib.Path(file).suffix == ".obj"]
        #     for j, (obj_name, obj_dict) in enumerate(self.tracked_objects.items()):
        #         obj_file = obj_name+".obj"
        #         assert(obj_file in meshlist)
        #         mesh = read_mesh(self.mesh_dir+"/"+obj_file)
        #         obj = { "mesh": mesh, "pose": obj_dict["pose"] }
        #         self.objects[0][j] = obj

        # else: raise ValueError(f"{self.mode} is an invalid mode")


        print("Camera loader initialized")

    def get_imagepaths_from_path(self, path, camera_name):

        # load images
        folders = [d for d in os.listdir(path) if os.path.isdir(path+"/"+d)]
        image_paths = {}
        for f in folders:
            if f in self.list_images:
                image_path = path+"/"+f+"/"+camera_name+".png"
                print(image_path)
                assert(os.path.isfile(image_path))
                image_paths[f]=image_path
        return image_paths

    def get_set_of_cameras_from_frame(self, idx):
        return self.cams[idx]

    def cam_static_generator(self):
        for i in range(self.n_cameras):
            yield self.get_camera_static(i)

    def cam_video_generator(self):
        for i in range(self.n_frames):
            yield self.get_set_of_cameras_from_frame(i)


        # # if len(self.objects)==1:
        # for obj in self.objects:
        #     for frame in obj['dict'].keys():
        #         frame['pose']
        #     print(obj)
        #     exit(1)
        #     plotter.plot_mesh(obj, size)
        # # else:
        # #     for i, frame in enumerate(self.cams):
        # #         for cam in frame:
        # #             plotter.plot_cam(cam, size, i)


    def normalize_wrt_aabb(self, aabb, side_length:float=1):

        # Load the bounding box or create it from the mesh vertices
        aabb = AABB(aabb.corners.cpu().numpy() if torch.is_tensor(aabb.corners) else aabb.corners)
        t = torch.from_numpy(-aabb.center).to(self.device)
        # print((np.float32(side_length / aabb.longest_extent)).shape)
        # print(s.shape)
        s = float(np.float32(side_length / aabb.longest_extent))

        # normalize camera pose
        collected_poses = []
        collected_intr = []
        for frame in self.cams:
            for camera in frame:
                if camera.pose not in collected_poses:
                    collected_poses.append(camera.pose)
                if camera.intr not in collected_intr:
                    collected_intr.append(camera.intr)
        for pose in collected_poses:
            pose.move_location(t)
            pose.uniform_scale(s*2,units="normalized")
        for intr in collected_intr:
            intr.uniform_scale(s*2,units="normalized")

        # normalize objects 
        collected_poses = []
        collected_meshes = []
        for frame in self.objects:
            for obj in frame:
                if obj["pose"] not in collected_poses:
                    collected_poses.append(obj["pose"])
                if obj["mesh"] not in collected_meshes:
                    collected_meshes.append(obj["mesh"])
        for pose in collected_poses:
            pose.move_location(t)
            pose.uniform_scale(s*2,units="normalized")
        for mesh in collected_meshes:
            mesh.uniform_scale(s*2,units="normalized")

    # visualization
    def visualize_images(self, image_name='rgb', show=True, save_path=None):
        images = []
        for frame in self.cams:
            for cam in frame:
                # cam.load_images()
                image = cam.get_image(image_name).detach().cpu().numpy()
                # wandb.log({"images": wandb.Image(image.numpy())})
                images.append(image)
        fig, axs = figures.create_mosaic_figure(images)

        if save_path is not None:
            plt.savefig(save_path)

        if show:
            plt.show()

        if not show:
            plt.close(fig)

    def visualize_overlayed_images(self, image_name='rgb', show=True, save_path=None ):

        images = []
        for frame, cams in enumerate(self.cams):
            # take just first object (TODO extend to multiple objects)
            obj = self.objects[frame][0]
            for cam in cams: 
                # cam.load_images()
                overlayed = cam.get_overlayed_image(obj, image_name).detach().cpu().numpy()
                images.append(overlayed)
        fig, axs = figures.create_mosaic_figure(images)

        if save_path is not None:
            plt.savefig(save_path)
            # wandb.log({"images overlayed": wandb.Image(save_path)})

        if show:
            plt.show()

        if not show:
            plt.close(fig)

    def plot_cams(self, size=0.1):
        for i, frame in enumerate(self.cams):
            for cam in frame:
                plotter.plot_cam(cam, size, i)
                # plotter.plot_frame(cam.pose, size, i)

    def plot_objects(self):
        for i, frame in enumerate(self.objects):
            for obj in frame:
                v = obj["mesh"].get_transformed_vertices(obj["pose"])
                # v = obj["mesh"].vertices
                plotter.plot_mesh(v,obj["mesh"].indices , frame=i)

    def show_frame_sequence(self, cam_id=0, image_name='rgb', wk=1):
        cams = list(map(list, zip(*self.cams)))
        for cam in cams[cam_id]:
            # cam.load_images([image_name])
            cam.show_image(image_name, wk)

    #Sampler
    def get_cam( self, cam_id, frame:int = 0):
        cam = self.cams[frame][cam_id]
        return cam

    def sample_rand_cams(self, n_cams:int = 1):
        cameras = np.random.choice(self.cams_flatten, n_cams, replace=False)
        return cameras





# if __name__=="__main__":
#     cam_loader = CamLoader(current+"/data", "blender_1_simple",['rgb'] )

    # cam_loader.plot_cams()
    # plotter.show()

    # c = cam_loader.get_camera_static(0)
    # c.load_images()
    # c.show_images()

    # cam_loader.visualize_images('rgb')

    # cams = cam_loader.get_set_of_cameras_from_frame(3)
    # for c in cams: c.show_image("normals")

    # plotter.show()
    # # c = gen.getCamOnSphere(0)
    # # grid = c.get_pixel_grid()
    # # eps = c.pix2eps(grid)
    












