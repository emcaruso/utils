try:
    import bpy
except:
    1
from contextlib import contextmanager
import copy
import os
import subprocess
import sys
import torch
import numpy as np
from mathutils import Matrix, Vector

from utils_ema.camera_cv import *
from utils_ema.geometry_pose import Pose


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


# def launch_blender_script( blend_file, script_path):
#     os.system("blender "+ blend_file +" --background --python "+script_path)


def launch_blender_script(blend_file, script_path, arguments=[""]):
    args = " ".join(arguments)
    os.system(
        "blender "
        + blend_file
        + " --background --python "
        + script_path
        + " -- "
        + args
    )


def launch_blender_script_local(blend_file, script_path, arguments=[""]):
    # import bpy if not already imported
    if "bpy" not in locals():
        import bpy

    arg_str = " ".join(arguments)
    subprocess.call("python " + script_path + " " + arg_str, shell=True)


def get_blend_file(blend_dir):
    dir = blend_dir
    blend_files = [
        os.path.abspath(dir + "/" + f)
        for f in list(os.listdir(dir))
        if f[-6:] == ".blend"
    ]
    assert len(blend_files) == 1
    return blend_files[0]


def blender_camera_transform(obj):
    obj.rotation_euler[0] += math.pi


def delete_collection(collection_name):
    # Check if the collection exists
    if collection_name in bpy.data.collections:
        # Get the collection
        collection = bpy.data.collections[collection_name]

        # Unlink all objects from the collection
        objects = collection.objects
        while objects:
            # Remove the object from the scene
            bpy.data.objects.remove(objects[0], do_unlink=True)

        # Delete the collection
        bpy.data.collections.remove(collection)

        print(f"Collection '{collection_name}' and its objects have been deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist.")


def set_collection_hide_val(collection_name, val, ignore_errors=False):

    if not ignore_errors:
        # Check if the collection already exists
        if collection_name not in bpy.data.collections:
            raise ValueError(f"collection {collection_name} do not exists")

    c = bpy.data.collections[collection_name]
    c.hide_viewport = val


def create_collection(collection_name):

    # Check if the collection already exists
    if collection_name in bpy.data.collections:
        print(f"Collection '{collection_name}' already exists.")
        return

    # Create a new collection
    new_collection = bpy.data.collections.new(collection_name)

    # Link the collection to the current scene
    bpy.context.scene.collection.children.link(new_collection)

    print(f"Collection '{collection_name}' has been created.")


def create_collection_hard(collection_name):
    delete_collection(collection_name)
    create_collection(collection_name)


def insert_object_into_collection(collection_name, obj):
    # Check if the collection exists
    if collection_name not in bpy.data.collections:
        print(f"Collection '{collection_name}' does not exist.")
        return

    # Get the collection
    collection = bpy.data.collections[collection_name]

    # Unlink object from its old collection (if any)
    old_collection = obj.users_collection[0] if obj.users_collection else None
    if old_collection:
        old_collection.objects.unlink(obj)

    # Link object to the new collection
    collection.objects.link(obj)
    print(f"Object '{obj.name}' inserted into collection '{collection_name}'")


def clear_animation_data(obj):
    obj.animation_data_clear()


def insert_objects_into_collection_byname(collection_name, object_names):

    # Link objects to the collection
    for object_name in object_names:
        if object_name in bpy.data.objects:
            obj = bpy.data.objects[object_name]
            insert_object_into_collection(collection_name, obj)
            print(f"Object '{obj.name}' inserted into collection '{collection_name}'")
        else:
            print(f"Object '{obj.name}' does not exist.")


def collect_objects_in_collection(collection_name):
    # Check if the collection exists
    if collection_name not in bpy.data.collections:
        print(f"Collection '{collection_name}' does not exist.")
        return []

    # Get the collection
    collection = bpy.data.collections[collection_name]

    # Collect objects in the collection
    objects = [obj for obj in collection.objects]

    # Sort objects by name
    objects.sort(key=lambda obj: obj.name)

    return objects


def generate_camera_from_camcv(cam, name):
    cam.name = name
    camera_data = bpy.data.cameras.new(name=cam.name)
    camera_data.sensor_width = cam.intr.sensor_size[0] * 1000
    camera_object = bpy.data.objects.new(cam.name, camera_data)
    bpy.context.scene.render.resolution_x = cam.intr.resolution[0]
    bpy.context.scene.render.resolution_y = cam.intr.resolution[1]
    K = cam.intr.K_und
    K_pix = cam.intr.K_pix_und

    lens = ((K[0, 0] + K[1, 1]) / 2) * 1000

    # access the camera shift x and y
    camera_data.shift_x = -(K_pix[0, 2] - cam.intr.resolution[0] / 2) / (
        cam.intr.resolution[0]
    )
    camera_data.shift_y = (K_pix[1, 2] - cam.intr.resolution[1] / 2) / (
        cam.intr.resolution[1]
    )

    camera_data.lens = lens
    # extrinsics
    camera_object.matrix_world = Matrix(cam.pose.get_T().numpy())
    blender_camera_transform(camera_object)
    return camera_object, camera_data


def set_object_pose(obj, pose: Pose):
    scale = copy.deepcopy(obj.scale)
    obj.rotation_mode = pose.euler.convention
    obj.matrix_world = Matrix(pose.get_T().numpy())
    obj.scale = scale
    # obj.location = pose.location().numpy()
    # obj.rotation_euler = pose.euler.e.numpy()


# def generate_camera_from_camcv(cam, name):
#     cam.name = name
#     camera_data = bpy.data.cameras.new(name=cam.name)
#     camera_data.sensor_width = cam.intr.sensor_size[0]*1000
#     camera_object = bpy.data.objects.new(cam.name, camera_data)

#     bpy.context.scene.render.resolution_x = cam.intr.resolution[0]
#     bpy.context.scene.render.resolution_y = cam.intr.resolution[1]

#     K = cam.intr.K_und
#     fx = K[0,0]
#     fy = K[1,1]
#     if fx>fy:
#         aspect_y = fx/fy
#         lens= fx*1000
#         bpy.context.scene.render.pixel_aspect_y = aspect_y
#     else:
#         aspect_x = fy/fx
#         lens= fy*1000
#         bpy.context.scene.render.pixel_aspect_x = aspect_x
#     # if fx>fy:
#     #     aspect_x = fx/fy
#     #     lens= fy*1000
#     #     bpy.context.scene.render.pixel_aspect_x = aspect_x
#     # else:
#     #     aspect_y = fy/fx
#     #     lens= fx*1000
#     #     bpy.context.scene.render.pixel_aspect_y = aspect_y

#     camera_data.lens = lens
#     camera_object.matrix_world=Matrix(cam.pose.get_T().numpy())
#     blender_camera_transform( camera_object )
#     return camera_object, camera_data


# def generate_camera_from_intrinsics(cam_dict, name):
# camera_data = bpy.data.cameras.new(name=name)
# camera_data.sensor_width = 6.6
# camera_object = bpy.data.objects.new(name, camera_data)
# bpy.context.scene.collection.objects.link(camera_object)
# K = cam_dict['camera_matrix']
# fx = K[0,0]
# fy = K[1,1]
# cx = K[0,2]
# cy = K[1,2]
# res_x = int(cam_dict['img_width'])
# res_y = int(cam_dict['img_height'])

# resolution = torch.LongTensor([res_x,res_y])

# bpy.context.scene.render.resolution_x = res_x
# bpy.context.scene.render.resolution_y = res_y
# asp_ratio = res_x/res_y
# sw = camera_data.sensor_width
# lens_x = K[0,0]*sw/res_x
# lens_y = K[1,1]*sw/(res_y*asp_ratio)
# lens = (lens_x+lens_y)/2
# camera_data.lens = lens
# # camera_data.shift_x = -(cx - res_x/2)/(res_x)
# # camera_data.shift_y = (cy - res_y/2)/(res_y)
# # camera_data.shift_y = (cy - res_y/2)/(res_y)*(1/asp_ratio)
# return camera_object, camera_data


def generate_intrinsics_from_camera(cam):
    # NO OFFSET AS CAMERAS IN BLENDER ARE ASSUMED TO BE SYNTHETIC
    camera_data = cam.data
    res_x = bpy.context.scene.render.resolution_x
    res_y = bpy.context.scene.render.resolution_y
    resolution = torch.LongTensor([res_x, res_y])
    lens = camera_data.lens * 0.001
    sw = camera_data.sensor_width * 0.001
    sh = sw * (res_y / res_x)
    K = torch.eye(3)
    K[0, 0] = lens
    K[1, 1] = lens
    K[0, 2] = (res_x / 2) * (sw / res_x)
    K[1, 2] = (res_y / 2) * (sw / res_x)
    intrinsics = Intrinsics(
        K=K,
        resolution=resolution,
        sensor_size=torch.FloatTensor([sw, sh]),
        units="meters",
    )

    # K[0,0] = lens*0.001*(res_x/sw)
    # K[1,1] = lens*0.001*(res_x/sw)
    # K[0,2] = (res_x/2)
    # K[1,2] = (res_y/2)
    # intrinsics = Intrinsics(K=K, resolution=resolution, units='pixels')
    # camera_data.shift_x = -(cx - res_x/2)/(res_x)
    # camera_data.shift_y = (cy - res_y/2)/(res_y)*(1/asp_ratio)
    return intrinsics
