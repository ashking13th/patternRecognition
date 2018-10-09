import numpy as np
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-d", "--dest", required=True, help="Output data set location")
args = vars(ap.parse_args())
