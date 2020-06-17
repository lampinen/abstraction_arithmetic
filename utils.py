import numpy as np

def save_config(filename, config): 
    with open(filename, "w") as fout: 
        fout.write("key, value\n") 
        keys = sorted(config.keys()) 
        for key in keys: 
            fout.write(key + ", " + str(config[key]) + "\n") 

def untree_dicts(tree, join="-"): 
    """Given named tree defined by dicts, unpacks to a flat dict with values
    being leaves and keys being the keys traversed to reach that leaf"""

    result = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            v_result = untree_dicts(v, join=join)
            for k2, v2 in v_result.items():
                result[k+ join + k2] = v2
        else:
            result[k] = v
    return result


