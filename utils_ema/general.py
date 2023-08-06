import json
import argparse

def print_hash(string, offs=5):
    l = len(string)+ (offs+1)*2
    print("\n"+"#" * l)
    print("#"*offs+" "+ string +" "+"#"*offs)
    print("#" * l+"\n")


def dump_parsed_args( path, args ):
    with open( path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_parsed_args( path):
    with open(path, 'r') as f:
        dict = json.load(f)
    return dict

def loadNpz( filepath ):
    if os.path.exists(filepath):
        return np.load(filepath)
    return None

