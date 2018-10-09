import numpy as np
import os
import argparse
from imageToColorHistogram import imageToHistogram as toHisto

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-d", "--dest", required=True, help="Output data set location")
args = vars(ap.parse_args())

for root, dirs, files in os.walk(args["source"]):
    for f in files:
        print(os.path.relpath(os.path.join(root, f), "."))
        dataset = toHisto(args["source"], f, os.path.relpath(os.path.join(root, f), "."), args["dest"])
