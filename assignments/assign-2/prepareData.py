import numpy as np
import os
import argparse
import imageToColorHistogram as toHisto
import cellPrep as cp

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-d", "--dest", required=True, help="Output data set location")
ap.add_argument("-t", "--type", required=True, help="Input data type - 1-Image data to color histogram; 2-Cell data to 2-d features")
args = vars(ap.parse_args())

i = 0
for root, dirs, files in os.walk(args["source"]):
    for f in files:
        i += 1
        path = os.path.relpath(os.path.join(root, f), ".")
        target = os.path.relpath(os.path.join(root, os.path.splitext(f)[0]))
        if args["type"] == "1":
            toHisto.create(args["source"], f, path, args["dest"], target)
        elif args["type"] == "2":
            cp.create(args["source"], f, path, args["dest"], target)
