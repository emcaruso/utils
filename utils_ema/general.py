import json
import argparse
import cv2
import torch
import numpy as np

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

