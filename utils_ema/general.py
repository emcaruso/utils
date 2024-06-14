import json
import pickle
import argparse
import cv2
import torch
import numpy as np
import time
from screeninfo import get_monitors
import os
import importlib.util

##### IMPORT ######

def load_module_from_path(file_path, module_name):
    """ Load a module from a given file path without altering sys.path. """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Cannot load module from {file_path}")

def load_class_from_path(path, class_name):
    module_name = 'custom_module'  # This is an arbitrary name

    # Create a module spec from the file location
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Extract the class from the module
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"No class named {class_name} found in {path}.")
    return cls

def load_function_from_path(path, function_name):
    module_name = 'custom_module'  # This is an arbitrary name

    # Create a module spec from the file location
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Extract the function from the module
    func = getattr(module, function_name, None)
    if func is None:
        raise ImportError(f"No function named {function_name} found in {path}.")
    return func


##### OTHER #####

def get_monitor():
    for m in get_monitors():
        return m

##### PERFORMANCE ######

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

def timing_decorator_print(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print("execution time: ",execution_time)
        return result
    return wrapper

##### PRINT ######

def print_hash(string, offs=5):
    l = len(string)+ (offs+1)*2
    print("\n"+"#" * l)
    print("#"*offs+" "+ string +" "+"#"*offs)
    print("#" * l+"\n")

def select_file_in_folder(dirpath):
    file_names = os.listdir(dirpath)
    string = ""
    for s in [ str(i)+") "+name+"\n"  for i, name in enumerate(file_names)]:
        string += s
    o = input("select file by number, or type\n"+string)
    if o.isnumeric():
        chosen_file = file_names[int(o)]
    else:
        chosen_file = o
    return chosen_file


##### PARSER ######

# def dump_parsed_args( path, args ):
#     with open( path, 'w') as f:
#         json.dump(args.__dict__, f, indent=2)

def dump_dict( path, dict ):
    with open( path, 'w') as f:
        json.dump(dict, f, indent=2)

def dump_parsed_args( path, args ):
    return dump_dict( path, args.__dict__ )

def load_parsed_args_as_namespace( path):
    dict = load_parsed_args_as_dict( path)
    return argparse.Namespace(**dict)

def load_parsed_args_as_dict( path):
    with open(path, 'r') as f:
        dict = json.load(f)
    return dict

##### PICKLE

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        loaded_object = pickle.load(filepath)
        return loaded_object

##### NPZ ######

def loadNpz( filepath ):
    if os.path.exists(filepath):
        return np.load(filepath)
    return None

# USER INPUT

def ask_question( question , values ):
    while True:
        choice = input(question)
        if choice.lower() in values:
            return choice.lower()
        print("Invalid input.")


