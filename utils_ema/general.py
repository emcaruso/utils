import json
import pickle
import argparse
import cv2
import torch
import numpy as np
import time

##### PERFORMANCE ######

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

##### PRINT ######

def print_hash(string, offs=5):
    l = len(string)+ (offs+1)*2
    print("\n"+"#" * l)
    print("#"*offs+" "+ string +" "+"#"*offs)
    print("#" * l+"\n")

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

