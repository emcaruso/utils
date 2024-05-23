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
    from .objects import Object
    from .diff_renderer import Renderer
except:
    from camera_cv import *
    from geometry_pose import *
    from geometry_euler import *
    from figures import *
    from plot import *
    from mesh import Mesh, read_mesh, AABB
    from objects import Object
    from diff_renderer import Renderer


class Scene():
    def __init__(self, data_dir, data_dataset, list_images, frames=None, device='cpu', load_images=None, dtype=torch.float32, resolution_drop=1. ):
        self.device=device
        self.data_dir = data_dir+"/"+data_dataset
        if not os.path.isdir(self.data_dir): raise TypeError("the path: "+self.data_dir+" doesn't exists")
        if frames is None: self.mode = 'static'
        else: self.mode = 'video'
        self.mesh_dir = self.data_dir+"/meshes"
        self.name_dataset = data_dataset
        self.list_images = list_images
        self.resolution_drop = resolution_drop
        self.frames = frames
        self.dtype = dtype
        print("dataset path: ",self.data_dir)
        if os.path.isfile(self.data_dir+"/"+self.mode+"/data.npz"):
            self.data_type = "blender"
            self.init_blender(load_images)
        elif os.path.isfile(self.data_dir+"/"+self.mode+"/data_basler.npz"):
            self.data_type = "basler"
            self.init_basler(load_images)
        else:
            print("not valid scene")
        self.resize_pixels(resolution_drop=resolution_drop)

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
        elif self.mode == "video":
            list_frame_names = [ f for f in list(os.listdir(self.data_dir+"/video")) if os.path.isdir(self.data_dir+"/video/"+f) ]
            list_frame_names.sort()
            frames_available = [ int(f.split("_")[-1]) for f in list_frame_names ]
            if self.frames == "All":
                self.frames = frames_available
            elif len(self.frames) == 1:
                self.frames = [self.frames[0]]
            else:
                self.frames = range(self.frames[0],self.frames[1])
            for frame in self.frames: assert(frame in frames_available)

        self.n_frames = len(self.frames)
        self.n_cameras = len(self.camera_names)
        self.cams = [[None]*self.n_cameras for _ in range(self.n_frames)]
        for j, cam_arr in enumerate(list(self.npz.values())):
            for i, frame in enumerate(self.frames):
                cam = cam_arr.tolist()
                cam = cam.to(self.device)
                cam = cam.dtype(self.dtype)
                cam_name = "Cam_"+str(j).zfill(3)
                image_paths = self.get_imagepaths_from_path(self.data_dir+"/video/frame_"+str(frame).zfill(3), cam_name)
                new_cam = cam.clone(same_intr=True, same_pose=True, image_paths=image_paths, name=cam_name)
                if load_images is not None: new_cam.load_images(load_images)
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
        elif self.mode == "video":
            frames_available = [ int(f.split("_")[-1]) for f in list(os.listdir(self.data_dir+"/video")) if os.path.isdir(self.data_dir+"/video/"+f) ]
            frames_available.sort()
            if self.frames == "All":
                self.frames = frames_available
            elif len(self.frames) == 1:
                self.frames = [self.frames[0]]
            else:
                self.frames = range(self.frames[0],self.frames[1])
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
                cam = Camera_cv(intrinsics= cam_dict["intrinsics"], pose=pose, frame=frame, image_paths=image_paths, name=cam_name+"_"+str(frame), device=self.device, dtype=self.dtype) 
                if load_images is not None: cam.load_images(load_images)
                self.cams[i][j] = cam
        self.cams_flatten = [ cam for frame in self.cams for cam in frame]

        # load objects
        meshlist = [ file for file in os.listdir(self.mesh_dir) if pathlib.Path(file).suffix == ".obj"]
        for j, (mesh_name, obj_dict) in enumerate(self.tracked_objects.items()):
            mesh_file = mesh_name+".obj"
            mesh_path = os.path.join(self.mesh_dir,mesh_file)
            assert(mesh_file in meshlist)
            mesh = read_mesh(mesh_path, device=self.device)
            mesh.compute_connectivity()
            mesh = mesh.to(device=self.device)
            for i, frame in enumerate(self.frames):
                if isinstance(obj_dict["pose"], Pose): pose = obj_dict["pose"]
                else: pose = obj_dict["pose"][frame]
                # obj = Object(mesh=mesh, pose=pose.to(device=self.device))
                obj = Object(mesh=mesh, pose=pose, device=self.device)
                self.objects[i][j] = obj
        self.objects_flatten = [ obj for frame in self.objects for obj in frame]

        # check units
        units = self.cams[0][0].pose.units
        for frame in self.cams:
            for cam in frame:
                assert(cam.pose.units==units)
        for frame in self.objects:
            for obj in frame:
                assert(obj.pose.units==units)

        print("Camera loader initialized")

    def get_imagepaths_from_path(self, path, camera_name):

        # load images
        folders = [d for d in os.listdir(path) if os.path.isdir(path+"/"+d)]
        image_paths = {}
        for f in folders:
            if f in self.list_images:
                image_path = path+"/"+f+"/"+camera_name+".png"
                print("image path: "+image_path)
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

    def resize_pixels(self, resolution_drop=1.):

        collected_intr = []
        for frame in self.cams:
            for camera in frame:
                if camera.intr not in collected_intr:
                    collected_intr.append(camera.intr)
        for intr in collected_intr:
            intr.resize_pixels(resolution_drop=resolution_drop)

        for frame in self.cams:
            for camera in frame:
                camera.set_resolution_drop(resolution_drop)

    def normalize_wrt_aabb(self, aabb, side_length:float=1):

        # Load the bounding box or create it from the mesh vertices
        aabb = AABB(aabb.corners.cpu().numpy() if torch.is_tensor(aabb.corners) else aabb.corners)
        t = torch.from_numpy(-aabb.center).to(self.device)
        # print((np.float32(side_length / aabb.longest_extent)).shape)
        # print(s.shape)
        s = float(np.float32(side_length / aabb.longest_extent))

        # normalize camera pose
        collected_cam_poses = self.collect_cam_poses()
        collected_intr = self.collect_intrs()
        collected_meshes = self.collect_meshes()
        collected_obj_poses = self.collect_obj_poses()

        for pose in collected_cam_poses:
            pose.move_location(t)
            pose.uniform_scale(s*2,units="normalized")
        for intr in collected_intr:
            intr.uniform_scale(s*2,units="normalized")
        for pose in collected_obj_poses:
            pose.move_location(t)
            pose.uniform_scale(s*2,units="normalized")
        for mesh in collected_meshes:
            mesh.uniform_scale(s*2,units="normalized")

    def collect_cam_poses(self):
        collected_poses = []
        for frame in self.cams:
            for camera in frame:
                if camera.pose not in collected_poses:
                    collected_poses.append(camera.pose)
        return collected_poses
        
    def collect_intrs(self):
        collected_intr = []
        for frame in self.cams:
            for camera in frame:
                if camera.intr not in collected_intr:
                    collected_intr.append(camera.intr)
        return collected_intr

    def collect_meshes(self):
        collected_meshes = []
        for frame in self.objects:
            for obj in frame:
                if obj.mesh not in collected_meshes:
                    collected_meshes.append(obj.mesh)
        return collected_meshes
        
    def collect_obj_poses(self):
        collected_poses = []
        for frame in self.objects:
            for obj in frame:
                if obj.pose not in collected_poses:
                    collected_poses.append(obj.pose)
        return collected_poses
        
    def collect_materials(self):
        collected_materials = []
        for frame in self.objects:
            for obj in frame:
                if obj.material not in collected_materials:
                    collected_materials.append(obj.material)
        return collected_materials
        

    # visualization
    def visualize_images(self, image_name='rgb', show=True, save_path=None):
        images = []
        for frame in self.cams:
            for cam in frame:
                # cam.load_images()
                # image = cam.get_image(image_name).swapped().clone().cpu().numpy()
                image = cam.get_image(image_name).clone().cpu().numpy()
                # wandb.log({"images": wandb.Image(image.numpy())})
                images.append(image)
        fig, axs = figures.create_mosaic_figure(images)

        if save_path is not None:
            print("saving scene images: "+save_path)
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def visualize_overlayed_images(self, image_name='rgb', show=True, save_path=None ):

        images = []
        for frame, cams in enumerate(self.cams):
            # take just first object (TODO extend to multiple objects)
            obj = self.objects[frame][0]
            for cam in cams: 
                # cam.load_images()
                overlayed = cam.get_overlayed_image(obj, image_name).clone().cpu().numpy()
                images.append(overlayed)
        fig, axs = figures.create_mosaic_figure(images)

        if save_path is not None:
            print("saving scene images overlayed: "+save_path)
            plt.savefig(save_path)
            # wandb.log({"images overlayed": wandb.Image(save_path)})

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_cams(self, size=0.2):
        for i, frame in enumerate(self.cams):
            for cam in frame:
                plotter.plot_cam(cam, size, i)
                # plotter.plot_frame(cam.pose, size, i)

    def plot_objects(self):
        for i, frame in enumerate(self.objects):
            for obj in frame:
                v = obj.mesh.get_transformed_vertices(obj.pose)
                # v = obj.mesh.vertices
                plotter.plot_mesh(v,obj.mesh.indices , frame=i)

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

    def test_diffrast_static(self):
        # for each camera at frame 0
        for cam in self.cams[0]:
            # for each object at frame 0
            for obj in self.objects[0]:
                gbuffer = Renderer.diffrast(cam, obj, ['mask','position','normal'])
                m = gbuffer['mask']
                p = gbuffer['position']
                n = gbuffer['normal']
                idxs = m.nonzero()
                n = n[idxs[:,0],idxs[:,1],:]
                p = p[idxs[:,0],idxs[:,1],:]
                # print(p.shape)
                im = Image(img=gbuffer['normal'])
                # plotter.plot_ray(p[:10000,...],n[:10000,...], length=0.05)
                # im = Image(img=gbuffer['position'])
                im.show()
                plotter.plot_points(p)
                plotter.show()
                exit(1)
        exit(1)






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
    












