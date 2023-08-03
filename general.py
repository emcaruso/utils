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
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(path, 'r') as f:
        args.__dict__ = json.load(f)
    return args

