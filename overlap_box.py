import csv
import json
import argparse
import os
from PIL import Image, ImageDraw as D
import numpy as np

   
#477 674 ->h-80= 594 -- w / h -> 0.803
#490 634 -> h-20..  w / h -> 0.798 
def get_bbox_coords(c0, c1, h, w, width, height):
    x_min = (c0 - w / 2) * width
    y_min = (c1 - h / 2) * height
    x_max = (c0 + w / 2) * width
    y_max = (c1 + h / 2) * height
    return (x_min, y_min, x_max, y_max)


def get_bb(file_path, image):

    file = open(file_path)
    bb = []
    line = []
    type = []

    width, height = image.size 
    for i in file:
        line = i.split("\n")[0].split(" ")
        c0 = float(line[1])
        c1 = float(line[2])
        w = float(line[3])
        h = float(line[4])
        get_bbox_coords(c0, c1, h, w, width, height)
        bb.append(get_bbox_coords(c0, c1, h, w, width, height))
        type.append(int(line[0]))
    return bb, type
   
def draw_bb(bb, image, labels, path_save):
#def draw_bb(bb, path_image, labels, path_save):
    # [x0, y0, x1, y1]
    #image = Image.open(path_image)
    draw = D.Draw(image)
    for k in range(len(bb)):
        if labels[k] == 0:
           color = 'red'
        else:
           color = 'black'
        draw.rectangle(bb[k], outline = color, width = 1) 
    if path_save !='':
        image.save(path_save)
    else:
        image.show()
    
""" |----------------- box1 -----------------|
    |                                    |
    | |------------- box2 -------------| |
    | |                                | |
    | | |---------- intersection ------| |
 """
# Intersection over Union -> area of intersection divided by the area of union of the two boxes
def check_overlap(box1, box2):
        intersect_x0 = max(box1[0], box2[0])
        intersect_y0 = max(box1[1], box2[1])
        intersect_x1 = min(box1[2], box2[2])
        intersect_y1 = min(box1[3], box2[3])
        intersect_area = max(0, intersect_x1 - intersect_x0) * max(0, intersect_y1 - intersect_y0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        return iou


def marge_bb(bb,image,path_save, labels, coord):
  #  path_image = 'dataset/imgs/stat_21_2109.00464_vis/7.png' 
    tmp = []
    remove = []
    new_labes = []
    new_bb = []

    k = 0
    while k < (len(bb)): 
        j = k + 1 # parte da quello successivo
        x0 = bb[k][0]
        y0 = bb[k][1]
        x1 = bb[k][2]
        y1 = bb[k][3]
        marge_box = False
       
        while j < (len(bb)): 
            iou = check_overlap([x0,y0,x1,y1], bb[j]) 
            if iou > 0.0:               # si sovrappongono 
                #print("The boxes overlap.", iou, j, 'k', k)
                # new bounding box 
                x0 = min(bb[j][0], x0)
                y0 = min(bb[j][1], y0)
                x1 = max(bb[j][2], x1) 
                y1 = max(bb[j][3], y1)

                remove.append(j) # salvo indici di bb che hanno fatto merge
                #####TODO Stessa classe
                marge_box = True    
            j +=1

        if marge_box : 
            if [x0,y0,x1,y1] not in tmp:
                tmp.append([x0,y0,x1,y1])      
            marge_box = False
        else:
            tmp.append(bb[k])
        k +=1 
        
    remove = sorted(list(set(remove)))
    for i in range(len(tmp)):
         if i not in remove:
            new_bb.append(tmp[i])
            new_labes.append(labels[i])
    return new_bb, new_labes
  

def merge_all_image(folder):
    # txt e img devono avere stesso nome!
    path_images = folder + 'images/'
    path_txt = folder + 'labels/'

    dir_list = os.listdir(path_images) 
    num = 0
    path_save = folder + 'merge/'
    create_folder(path_save)

    for d in dir_list:

        path_image = path_images + d 
        name = '.'.join(d.split('.')[0:len(d.split('.'))-1])
        txt = path_txt + name + '.txt'
        
        save_image = path_save + d        
        new_bb, new_labels = get_bb_merge(path_image, txt, save_image)
        num +=1


def sort_data(bb, labels, index):
    combined = list(zip(bb, labels))
    sorted_combined = sorted(combined, key=lambda x: x[0][index])
    sorted_bb, sorted_labels = zip(*sorted_combined)
    return sorted_bb, sorted_labels


def get_bb_merge(path_image, txt, path_save):
    image = Image.open(path_image)
    bb, labels = get_bb(txt,image)

    sorted_bb_y, sorted_labels_y  = sort_data(bb, labels, 1)
    bb_y, labels_y = marge_bb(list(sorted_bb_y), image, '', list(sorted_labels_y), 'y') #y

    sorted_bb_x, sorted_labels_x = sort_data(bb_y, labels_y, 0)
    new_bb, new_labels = marge_bb(list(sorted_bb_x), image, '', list(sorted_labels_x), 'x') #x
    if path_save != '':
        draw_bb(new_bb, image, new_labels, path_save)
    return new_bb, new_labels


def create_folder(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
    #folder = 'exp_test/'
    folder = 'C:/Users/ninad/Desktop/ACL_P10-1160_exp2/'
    merge_all_image(folder)

