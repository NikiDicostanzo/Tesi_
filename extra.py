import csv
import json
import argparse
import os
from PIL import Image, ImageDraw as D
import numpy as np
import shutil


def all_image(folder):
    path_json = 'dataset/test.json'
    with open(path_json, errors="ignore") as json_file:
        j = json.load(json_file)

        for m in range(len(j)): #ciclo sui doc
            img_pdf = j[m]["imgs_path"]
            #print(img_pdf)
            for i in img_pdf:
                print(i)
                name = i.split('/')
                new = 'image_test/' + name[len(name)-2] + '_' + name[len(name)-1]
                print(new)
                shutil.copy(i, new)
        
   



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
    folder = 'exp2/'
    all_image(folder)
    #marge_bb()