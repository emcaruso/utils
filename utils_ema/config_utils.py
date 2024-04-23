from omegaconf import OmegaConf, DictConfig, ListConfig
import omegaconf
import os
import re

def load_yaml(path):
    config = OmegaConf.load(path)

    # environment variable
    os.environ['YAML_DIR'] = os.path.dirname(path)
    OmegaConf.resolve(config)

    def apply_abspath_to_paths(config, substring):
        # Funzione per applicare os.path.abspath a tutti i valori di configurazione
        # che contengono una certa substring

        def recurse_and_apply(cfg):
            if isinstance(cfg, DictConfig):
                for key, value in cfg.items():
                    if isinstance(value, str) and substring in value:
                        cfg[key] = os.path.abspath(value)  # Applica abspath se la condizione è soddisfatta
                    elif isinstance(value, (DictConfig, ListConfig)):
                        recurse_and_apply(value)  # Ricorsione per sottodict e liste
            elif isinstance(cfg, ListConfig):
                for idx, item in enumerate(cfg):
                    if isinstance(item, str) and substring in item:
                        cfg[idx] = os.path.abspath(item)  # Applica abspath alle stringhe nella lista
                    elif isinstance(item, (DictConfig, ListConfig)):
                        recurse_and_apply(item)  # Ricorsione per sottodict e liste

        recurse_and_apply(config)  # Avvia la ricorsione dalla radice della configurazione


    apply_abspath_to_paths(config, os.environ['YAML_DIR'])

    return config
