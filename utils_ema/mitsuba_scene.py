import mitsuba as mi
from omegaconf import OmegaConf
import os
from pathlib import Path
from utils_ema.text import *
import torch
import numpy as np
import drjit as dr
from utils_ema.general import timing_decorator_print, timing_decorator
import xmltodict

# from pprint import pp as print


class MitsubaScene:

    mitsuba_blender_ratio_intensity = 4 * np.pi

    def __init__(self, xml_path, max_depth=None, rr_depth=None):
        self.xml_path = xml_path
        self.xml_dir = os.path.abspath(os.path.dirname(xml_path))
        self.max_depth = max_depth
        self.rr_depth = rr_depth
        self.parameters = None
        self.scene = None
        self.replace_dict = None

    def replace_parameters(self, replace_dict):
        self.replace_dict = replace_dict

    def load_scene(self):

        # self.scene = mi.load_file(self.xml_path)

        from mitsuba import ScalarTransform4f as T

        def preprocess(main_dict):
            items_to_add = []
            keys_to_remove = []

            del main_dict["default"]

            for el_type, item in main_dict.items():
                key, val = None, None

                if el_type == "integrator":
                    key = el_type
                    val = {
                        "type": item["type"],
                        "id": key,
                        "max_depth": int(item["integer"][0]["value"]),
                        "rr_depth": int(item["integer"][1]["value"]),
                    }
                    # key = "integrator"
                    items_to_add.append((key, val))
                    # keys_to_remove.append( key)

                elif el_type == "sensor":

                    if item.__class__ != list:
                        item = [item]
                    for i, d in enumerate(item):
                        t = [
                            float(n)
                            for n in d["transform"]["translate"]["value"].split()
                        ]
                        rx = float(d["transform"]["rotate"][0]["angle"])
                        ry = float(d["transform"]["rotate"][1]["angle"])
                        rz = float(d["transform"]["rotate"][2]["angle"])

                        key = "sensor_" + str(i)
                        val = {
                            "type": d["type"],
                            "id": key,
                            "near_clip": float(d["float"][3]["value"]),
                            "far_clip": float(d["float"][4]["value"]),
                            "fov": float(d["float"][0]["value"]),
                            "fov_axis": d["string"]["value"],
                            "principal_point_offset_x": float(d["float"][1]["value"]),
                            "principal_point_offset_y": float(d["float"][2]["value"]),
                            "to_world": T.translate(t)
                            .rotate([0, 0, 1], rz)
                            .rotate([0, 1, 0], ry)
                            .rotate([1, 0, 0], rx),
                            "film": {
                                "type": d["film"]["type"],
                                "width": int(d["film"]["integer"][0]["value"]),
                                "height": int(d["film"]["integer"][1]["value"]),
                                "pixel_format": "rgb",
                            },
                        }
                        items_to_add.append((key, val))
                    keys_to_remove.append(el_type)

                elif el_type == "bsdf":

                    if item.__class__ != list:
                        item = [item]
                    for i, d in enumerate(item):
                        args = []

                        if "texture" in list(d["bsdf"].keys()):
                            textures = d["bsdf"]["texture"]
                            if textures.__class__ != list:
                                textures = [textures]
                            for texture in textures:
                                args.append(
                                    (
                                        texture["name"],
                                        {
                                            "type": "bitmap",
                                            "filename": os.path.join(
                                                self.xml_dir, texture["string"]["value"]
                                            ),
                                        },
                                    )
                                )

                        if "rgb" in list(d["bsdf"].keys()):
                            rgbs = d["bsdf"]["rgb"]
                            if rgbs.__class__ != list:
                                rgbs = [rgbs]
                            for rgb in rgbs:
                                args.append(
                                    (
                                        rgb["name"],
                                        {
                                            "type": "rgb",
                                            "value": [
                                                float(v) for v in rgb["value"].split()
                                            ],
                                        },
                                    )
                                )

                        if "float" in list(d["bsdf"].keys()):
                            floats = d["bsdf"]["float"]
                            if floats.__class__ != list:
                                floats = [floats]
                            for fl in floats:
                                args.append((fl["name"], float(fl["value"])))

                        key = d["id"]

                        val_bsdf = {"type": d["bsdf"]["type"]}
                        for arg in args:
                            val_bsdf[arg[0]] = arg[1]

                        # val = {"type": d["type"], "id": key+"_ref", "bsdf":val_bsdf}
                        val = {"type": d["type"], "id": d["id"], "bsdf": val_bsdf}

                        # val = {"type": d["bsdf"]["type"], "id": key}
                        # for arg in args:
                        #     val[arg[0]] = arg[1]

                        items_to_add.append((key, val))
                    keys_to_remove.append(el_type)

                elif el_type == "emitter":

                    if item.__class__ != list:
                        item = [item]
                    for i, d in enumerate(item):

                        key = "emitter_" + str(i)
                        val = {
                            "type": d["type"],
                            "id": key,
                            "position": [
                                float(d["point"]["x"]),
                                float(d["point"]["y"]),
                                float(d["point"]["z"]),
                            ],
                            "intensity": {
                                "type": "rgb",
                                "value": [float(v) for v in d["rgb"]["value"].split()],
                            },
                        }

                        items_to_add.append((key, val))
                    keys_to_remove.append(el_type)

                elif el_type == "shape":

                    if item.__class__ != list:
                        item = [item]
                    for i, d in enumerate(item):

                        key = Path(d["string"]["value"]).stem

                        val = {
                            "type": d["type"],
                            "id": key,
                            "filename": os.path.join(
                                self.xml_dir, d["string"]["value"]
                            ),
                            "ref": {"type": "ref", "id": d["ref"]["id"]},
                        }

                        items_to_add.append((key, val))
                    keys_to_remove.append(el_type)

            for k in keys_to_remove:
                del main_dict[k]

            for i in items_to_add:
                if i[0] is not None:
                    main_dict[i[0]] = i[1]

            if self.replace_dict is not None:
                mdk = list(main_dict.keys())
                for k, v in self.replace_dict.items():
                    print(k, v)
                    main_dict[k] = OmegaConf.to_container(v)

        with open(self.xml_path, "r") as file:
            xml_data = file.read()
            dict_data = xmltodict.parse(xml_data, attr_prefix="")
            dict_data = dict_data["scene"]
            del dict_data["version"]
            keys_to_remove = []
            items_to_append = []
            for key, item in dict_data.items():
                if key == "default":
                    continue
                if item.__class__ == list:
                    for i, it in enumerate(item):
                        items_to_append.append((key + "_" + str(i).zfill(4), it))
                keys_to_remove.append(key)

        preprocess(dict_data)

        dict_data["type"] = "scene"
        self.scene = mi.load_dict(dict_data)

    def set_depths(self, max_depth=None, rr_depth=None):

        if max_depth is not None:
            self.max_depth = max_depth
        if rr_depth is not None:
            self.rr_depth = rr_depth

        content = read_content(self.xml_path)
        lines = content.splitlines()

        # set max_depth
        if self.max_depth is not None:
            lines_md = find_lines_with_string(self.xml_path, "max_depth")
            assert len(lines_md) == 1
            line_old = lines_md[0][1]
            line_id = lines_md[0][0]
            tkn = '<integer name="max_depth" value="'
            to_sub = line_old.split(tkn)[-1].split('"')[0]
            lines[line_id] = lines[line_id].replace(to_sub, str(self.max_depth))

        # set rr_depth
        if self.rr_depth is not None:
            lines_rr = find_lines_with_string(self.xml_path, "rr_depth")
            if len(lines_rr) == 0:
                line_id = line_id + 1
                line_old = line_old.replace("max_depth", "rr_depth")
                line_insert = line_old.replace(to_sub, str(rr_depth))
                lines.insert(line_id, line_insert)
            elif len(lines_rr) == 1:
                line_old = lines_rr[0][1]
                line_id = lines_rr[0][0]
                tkn = '<integer name="rr_depth" value="'
                to_sub = line_old.split(tkn)[-1].split('"')[0]
                lines[line_id] = lines[line_id].replace(to_sub, str(self.rr_depth))
            else:
                raise ValueError(
                    f"multiple line with rr_depth in xml file {self.xml_path}"
                )

        content_new = ""
        for line in lines:
            content_new += line + "\n"

        write_content(self.xml_path, content_new)

    def get_params(self):
        if self.scene is None:
            self.load_scene()
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

    def param_is_light_intensity(self, param):
        return param[0].split(".")[-2] == "intensity"

    def get_intensity_colors_from_intensity_parameters(self):
        params_list = []
        for param in self.parameters:
            if self.param_is_light_intensity(param):
                intensity_mits = param[1].torch()
                max_val = intensity_mits.max()
                intensity = max_val * self.mitsuba_blender_ratio_intensity
                color = intensity_mits / max_val
                params_list.append({"color": color, "intensity": intensity})
        return params_list

    # def get_pointlight_idxs_on(self):
    #     c = 0
    #     idxs = []
    #     for param in self.parameters:
    #         if self.param_is_light_intensity(param):
    #             intensity = self.parameters[param[0]]
    #             if dr.any(intensity > 0)[0]:
    #                 idxs.append(c)
    #             c += 1
    #     return idxs
    #
    # def get_pointlight_intensity_from_idx(self, pointlight_idx):
    #     name = self.get_pointlight_name_from_idx(pointlight_idx)
    #     return self.parameters[name + ".intensity.value"]

    def get_pointlight_name_from_idx(self, pointlight_idx):
        c = 0
        for param in self.parameters:
            if self.param_is_light_intensity(param):
                if c == pointlight_idx:
                    return param[0].split(".")[0]
                c += 1

    def set_pointlight_intensity_from_idx(self, pointlight_idx, light):
        c = 0
        intensity = (
            light.intensity * light.color
        ) / self.mitsuba_blender_ratio_intensity
        for param in self.parameters:
            if self.param_is_light_intensity(param):
                if c == pointlight_idx:
                    self.parameters[param[0]] = self.parameters[param[0]].__class__(
                        intensity.tolist()
                    )
                    break
                c += 1
        # self.parameters.update()

    # def set_pointlight_intensity(self, obj_name, intensity):
    #     if self.parameters is None:
    #         raise ValueError("Parameters not initialized (traverse mi scene)")
    #     if torch.is_tensor(intensity):
    #         intensity = intensity.numpy()
    #     assert len(intensity) == 3
    #     intensity[[0, 2]] = intensity[[2, 0]]  # rgb to bgr
    #     self.parameters[obj_name + ".intensity.value"] = intensity
    #     # self.parameters.update()

    def translate_obj(self, obj_name, offset):

        if torch.is_tensor(offset):
            offset = offset.numpy()

        if self.parameters is None:
            raise ValueError("Parameters not initialized (traverse mi scene)")

        assert len(offset.shape) == 1
        assert len(offset) == 3

        # manage offset
        offset[[1, 2]] = offset[[2, 1]]
        offset[2] *= -1
        offset = np.expand_dims(offset, axis=0)

        # v = self.parameters[obj_name+".vertex_positions"]
        # v_torch = v.numpy().reshape([-1,3])
        # v_torch += offset
        # self.parameters[obj_name+".vertex_positions"] = v_torch.flatten()

        vertices = dr.unravel(
            mi.Point3f, self.parameters[obj_name + ".vertex_positions"]
        )
        self.parameters[obj_name + ".vertex_positions"] = dr.ravel(
            mi.Transform4f.translate(offset) @ vertices
        )

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

        # spec_sum = 0
        # for i in range(spp):
        ## for i in range(8):
        #    # ray.scale_differential(dr.rsqrt(spp))
        #    # self.scene_mitsuba.scene.ray_intersect(ray)
        #    spec, mask, aov = integrator.sample(scene=self.scene, ray=ray, sampler=sampler)
        #    spec_sum = spec + spec_sum
        #        # res = spec + res
        #    sampler.advance()
        # res = spec_sum/spp

        if optim:
            dr.set_flag(dr.JitFlag.LoopRecord, True)
            # dr.ad.GradScope(active=True)

        return res, mask, aov

    # def render_dr_rays(self, ray:mi.RayDifferential3f):


#     @dr.wrap_ad(source='torch', target='drjit')
#     def render_rays_torch(self, origin, cam_dir, spp=32, seed=42):
#         return self.render_rays(origin, cam_dir, spp=spp, seed=seed)
