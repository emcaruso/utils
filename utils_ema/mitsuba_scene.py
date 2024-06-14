import mitsuba as mi
from utils_ema.text import *
import torch
import numpy as np
import drjit as dr
from utils_ema.general import timing_decorator_print, timing_decorator


class MitsubaScene():

    def __init__(self, xml_path, max_depth=None, rr_depth=None):
        self.xml_path = xml_path
        self.max_depth = max_depth
        self.rr_depth = rr_depth
        self.parameters = None
        self.scene = None

    def load_scene(self):
        self.scene = mi.load_file(self.xml_path)

    def save_lights(self):
        # self.emitters = self.scene.emitters().copy()
        # self.scene.emitters().clear()
        pass

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

    def get_cam(self, cam_idx):
        return self.scene.sensors()[cam_idx]

    def get_cams(self):
        return self.scene.sensors()

    def get_lights(self):
        return self.scene.emitters()

    def get_light(self, light_idx):
        return self.scene.emitters()[light_idx]

    def get_pointlight_idxs_on(self):
        c = 0
        idxs = []
        for param in self.parameters:
            if param[0].split('.')[-2]=="intensity":
                intensity = self.parameters[param[0]]
                if dr.any(intensity > 0)[0]:
                    idxs.append(c)
                c += 1
        return idxs

    def get_pointlight_intensity_from_idx(self, pointlight_idx):
        c = 0
        for param in self.parameters:
            if param[0].split('.')[-2]=="intensity":
                if c==pointlight_idx:
                    return self.parameters[param[0]]
                c += 1
        # self.parameters.update()

    def set_pointlight_intensity_from_idx(self, pointlight_idx, intensity):
        c = 0
        for param in self.parameters:
            if param[0].split('.')[-2]=="intensity":
                if c==pointlight_idx:
                    self.parameters[param[0]]=intensity*100
                c += 1
        # self.parameters.update()

    def set_pointlight_intensity(self, obj_name, intensity):
        if self.parameters is None: raise ValueError("Parameters not initialized (traverse mi scene)")
        if torch.is_tensor(intensity): intensity = intensity.numpy()
        assert(len(intensity)==3)
        intensity[[0,2]] = intensity[[2,0]] #rgb to bgr
        self.parameters[obj_name+".intensity.value"] = intensity
        # self.parameters.update()

    def translate_obj(self, obj_name, offset):

        if torch.is_tensor(offset): offset = offset.numpy()

        if self.parameters is None: raise ValueError("Parameters not initialized (traverse mi scene)")

        assert(len(offset.shape)==1)
        assert(len(offset)==3)

        # manage offset
        offset[[1,2]] = offset[[2,1]]
        offset[2] *= -1
        offset = np.expand_dims(offset, axis=0)

        v = self.parameters[obj_name+".vertex_positions"]
        v_torch = v.numpy().reshape([-1,3])
        v_torch += offset

        self.parameters[obj_name+".vertex_positions"] = v_torch.flatten()
        # self.parameters.update()


    # @dr.wrap_ad(source='drjit',target="torch")
    # def render_rays(self, origin:np.ndarray, cam_dir:np.ndarray, seed=42, spp=32, optim=False):
    # @timing_decorator_print
    def render_rays(self, ray, seed=42, spp=32, optim=False):

        if optim:
            dr.set_flag(dr.JitFlag.LoopRecord, False)
            # dr.ad.GradScope(active=False)


        # sampler
        n = dr.shape(ray.o)[1]
        sampler = self.scene.sensors()[0].sampler().fork()
        wavefront_size = n
        sampler.set_sample_count(spp)
        sampler.set_samples_per_wavefront(spp)
        sampler.seed(seed, wavefront_size)
        integrator = self.scene.integrator()

        ###############

        spec, mask, aov = integrator.sample(scene=self.scene, ray=ray, sampler=sampler)
        res = spec

        #####################

        #spec_sum = 0
        #for i in range(spp):
        ## for i in range(8):
        #    # ray.scale_differential(dr.rsqrt(spp))
        #    # self.scene_mitsuba.scene.ray_intersect(ray)
        #    spec, mask, aov = integrator.sample(scene=self.scene, ray=ray, sampler=sampler)
        #    spec_sum = spec + spec_sum
        #        # res = spec + res
        #    sampler.advance()
        #res = spec_sum/spp



        if optim:
            dr.set_flag(dr.JitFlag.LoopRecord, True)
            # dr.ad.GradScope(active=True)


        return res, mask, aov

    # def render_dr_rays(self, ray:mi.RayDifferential3f):

#     @dr.wrap_ad(source='torch', target='drjit')
#     def render_rays_torch(self, origin, cam_dir, spp=32, seed=42):
#         return self.render_rays(origin, cam_dir, spp=spp, seed=seed)
