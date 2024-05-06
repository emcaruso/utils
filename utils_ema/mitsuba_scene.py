import mitsuba as mi
from utils_ema.text import *
import torch
import numpy as np
import drjit as dr


class MitsubaScene():

    def __init__(self, xml_path, max_depth=None, rr_depth=None):
        self.xml_path = xml_path
        self.max_depth = max_depth
        self.rr_depth = rr_depth
        self.parameters = None
        self.scene = None

    def load_scene(self):
        self.scene = mi.load_file(self.xml_path)

    def set_depths(self, max_depth=None, rr_depth=None):

        if max_depth is not None: self.max_depth=max_depth
        if rr_depth is not None: self.rr_depth=rr_depth

        content = read_content(self.xml_path)
        lines = content.splitlines()

        #set max_depth
        if self.max_depth is not None:
            lines_md = find_lines_with_string(self.xml_path, "max_depth")
            assert(len(lines_md)==1)
            line_old = lines_md[0][1]
            line_id = lines_md[0][0]
            tkn = '<integer name="max_depth" value="'
            to_sub = line_old.split(tkn)[-1].split('"')[0]
            lines[line_id] = lines[line_id].replace(to_sub, str(self.max_depth))

        #set rr_depth
        if self.rr_depth is not None:
            lines_rr = find_lines_with_string(self.xml_path, "rr_depth")
            if (len(lines_rr)==0):
                line_id = line_id+1
                line_old = line_old.replace("max_depth","rr_depth")
                line_insert = line_old.replace(to_sub,str(rr_depth))
                lines.insert(line_id, line_insert)
            elif (len(lines_rr)==1):
                line_old = lines_rr[0][1]
                line_id = lines_rr[0][0]
                tkn = '<integer name="rr_depth" value="'
                to_sub = line_old.split(tkn)[-1].split('"')[0]
                lines[line_id] = lines[line_id].replace(to_sub, str(self.rr_depth))
            else:
                raise ValueError(f"multiple line with rr_depth in xml file {self.xml_path}")

        content_new = ""
        for line in lines: content_new += line+"\n"

        write_content(self.xml_path, content_new)

    def get_params(self):
        self.parameters = mi.traverse(self.scene)
        return self.parameters

    def print_params(self):
        print(self.get_params())

    def translate_obj(self, obj_name, offset):

        if torch.is_tensor(offset): offset = offset.numpy()

        assert(len(offset.shape)==1)
        assert(len(offset)==3)

        # manage offset
        offset[[1,2]] = offset[[2,1]]
        offset[2] *= -1
        offset = np.expand_dims(offset, axis=0)

        self.get_params()
        
        # print(self.parameters[obj_name+".vertex_positions"])

        v = self.parameters[obj_name+".vertex_positions"]
        v_torch = v.numpy().reshape([-1,3])
        v_torch += offset

        # print(v_torch.shape)
        # print(offset.shape)

        self.parameters[obj_name+".vertex_positions"] = v_torch.flatten()
        self.parameters.update()



        # print(v_torch)
        # print(self.parameters[obj_name+".vertex_positions"])
        # exit(1)

    def render_rays(self, origin:np.ndarray, cam_dir, seed=42, spp=32):

        # get rays
        if torch.torch.is_tensor(origin): origin = origin.numpy()
        if torch.torch.is_tensor(cam_dir): cam_dir = cam_dir.numpy()
        n = origin.shape[0]
        origin = np.expand_dims(origin,0)
        cam_dir =  np.expand_dims(cam_dir,0)
        origin = origin.repeat(spp, 0)
        cam_dir = cam_dir.repeat(spp, 0)
        origin = origin.reshape( [-1, 3] )
        cam_dir = cam_dir.reshape( [-1, 3] )

        # sampler
        sampler = self.scene.sensors()[0].sampler().fork()
        spp_per_pass = spp
        n_passes = int(spp/spp_per_pass)
        wavefront_size = int(n * spp_per_pass)
        sampler.set_sample_count(spp)
        sampler.set_samples_per_wavefront(spp_per_pass)
        sampler.seed(seed, wavefront_size)

        # import ipdb; ipdb.set_trace()

        origin = dr.cuda.Array3f(origin)
        cam_dir = dr.cuda.Array3f(cam_dir)
        ray = mi.RayDifferential3f(o=origin, d=cam_dir)
        ray.scale_differential(dr.rsqrt(spp))

        # self.scene_mitsuba.scene.ray_intersect(ray)
        integrator = self.scene.integrator()

        spec, mask, aov = integrator.sample(scene=self.scene, ray=ray, sampler=sampler)
        return spec, mask, aov

    @dr.wrap_ad(source='torch', target='drjit')
    def render_rays_torch(self, origin, cam_dir, spp=32, seed=42):
        return self.render_rays(origin, cam_dir, spp=spp, seed=seed)


    def set_pointlight_intensity(self, obj_name, intensity):
        if torch.is_tensor(intensity): intensity = intensity.numpy()
        assert(len(intensity)==3)
        intensity[[0,2]] = intensity[[2,0]] #rgb to bgr
        self.get_params()
        self.parameters[obj_name+".intensity.value"] = intensity
