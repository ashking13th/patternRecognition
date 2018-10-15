import numpy as np
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
args = vars(ap.parse_args())

i=0
for root, dirs, files in os.walk(args["source"]):
    for f in files:
        i += 1
        print("Image No. : ", i)
        path = os.path.relpath(os.path.join(root, f), ".")
        target = os.path.relpath(os.path.join(root, os.path.splitext(f)[0]))
        print(target)
        # toHisto.create(args["source"], f, path, args["dest"], target)
