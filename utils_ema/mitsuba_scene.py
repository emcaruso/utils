import mitsuba as mi
from utils_ema.text import *
import torch

class MitsubaScene():

    def __init__(self, xml_path, max_depth, rr_depth):
        self.xml_path = xml_path
        self.max_depth = max_depth
        self.rr_depth = rr_depth
        self.parameters = None

    def load_scene(self):
        self.scene = mi.load_file(self.xml_path)

    def set_depths(self):
        content = read_content(self.xml_path)
        lines = content.splitlines()

        #set max_depth
        lines_md = find_lines_with_string(self.xml_path, "max_depth")
        assert(len(lines_md)==1)
        line_old = lines_md[0][1]
        line_id = lines_md[0][0]
        tkn = '<integer name="max_depth" value="'
        to_sub = line_old.split(tkn)[-1].split('"')[0]
        lines[line_id] = lines[line_id].replace(to_sub, str(self.max_depth))

        #set rr_depth
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
            raise ValueError("")

        content_new = ""
        for line in lines: content_new += line+"\n"

        write_content(self.xml_path, content_new)

    def get_params(self):
        if self.parameters is None:
            self.parameters = mi.traverse(self.scene)
        return self.parameters

    def print_params(self):
        print(self.get_params())

    def translate_obj(self, obj_name, offset):

        if torch.is_tensor(offset): offset = offset.numpy()
        assert(len(offset)==3)
        # offset.flip(
        self.get_params()
        v = self.parameters[obj_name+".vertex_positions"]
        v_torch = v.numpy().reshape([-1,3])
        offset[[1,2]] = offset[[2,1]]
        offset[2] *= -1
        v_torch += offset
        self.parameters[obj_name+".vertex_positions"] = v_torch.flatten()
        self.parameters.update()

    def set_pointlight_intensity(self, obj_name, intensity):
        if torch.is_tensor(intensity): intensity = intensity.numpy()
        assert(len(intensity)==3)
        intensity[[0,2]] = intensity[[2,0]] #rgb to bgr
        self.get_params()
        self.parameters[obj_name+".intensity.value"] = intensity
