import csv
import json
import argparse
import os
from PIL import Image, ImageDraw as D
import numpy as np

from overlap_box import merge_all_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
    folder = 'exp_test/'
    merge_all_image(folder) # merge bounding box

    