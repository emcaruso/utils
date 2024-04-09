from omegaconf import OmegaConf
import os

def load_yaml(path):
    cfg = OmegaConf.load(path)

    if "paths" in cfg:
        yaml_dir = os.path.dirname(path)
        for k,v in cfg.paths.items():
            cfg.paths[k] = os.path.abspath(os.path.join(yaml_dir, v))

    return cfg
