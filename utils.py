import numpy as np

def save_config(filename, config): 
    with open(filename, "w") as fout: 
	fout.write("key, value\n") 
	keys = sorted(config.keys()) 
	for key in keys: 
	    fout.write(key + ", " + str(config[key]) + "\n") 


