import csv
import json
import argparse
import os
from PIL import Image, ImageDraw as D
import numpy as np


# Covert w, h, and center to bb
def get_bbox_coords(c0, c1, h, w, width, height):
    x_min = (c0 - w / 2) * width
    y_min = (c1 - h / 2) * height
    x_max = (c0 + w / 2) * width
    y_max = (c1 + h / 2) * height
    return (x_min, y_min, x_max, y_max)

# Get bb from labels of detection
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
   

def draw_bb(bb, image, labels):
    draw = D.Draw(image)
    for k in range(len(bb)):
        if labels[k] == 0:
           color = 'red'
        else:
           color = 'black'
        draw.rectangle(bb[k], outline = color, width = 1) 


def marge_bb(bb, labels):
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
            if iou > 0.0 and labels[k] == labels[j]:               # si sovrappongono 
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
            # elfi # aggiugnere se si sopvrappongono e classi diverse di prendere quella con confidenza >
        if marge_box : 
            #if [x0,y0,x1,y1] not in tmp:
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

# Merge su una immagine
def get_bb_merge(path_image, txt, path_save, detect):
    image = Image.open(path_image)
    bb, labels = get_bb(txt,image)

    sorted_bb_y, sorted_labels_y  = sort_data(bb, labels, 1)
    bb_y, labels_y = marge_bb(list(sorted_bb_y), list(sorted_labels_y)) #y

    sorted_bb_x, sorted_labels_x = sort_data(bb_y, labels_y, 0)
    new_bb, new_labels = marge_bb(list(sorted_bb_x), list(sorted_labels_x)) #x
    draw_bb(new_bb, image, new_labels)
    if path_save != '':
        if detect != '': ## Per visualizzare in un'unica immagine
            ###
            image_detect = Image.open(detect) # immagine di yolo
            dst = get_concat_h(image, image_detect)  
            dst.save(path_save)
        else:
            image.save(path_save)
    #else:
    #    image.show()
    return new_bb, new_labels

# Merge su tutte le immagini
def merge_all_image(folder, type_img):
    # txt e img devono avere stesso nome!
    path_images = folder + 'images/'
    path_txt = folder + 'labels/'
    images_detect = folder + 'detect/'

    dir_list = os.listdir(path_images) 
    num = 0
    path_save = folder + 'merge/'
    create_folder(path_save)

    path_json_save = folder + 'json_yolo/'
    create_folder(path_json_save)
    save_bb =[]

    save_name = True
    ind = 0
    data = []
    for d in dir_list:

        path_image = path_images + d 
        name = get_name(d)
        txt = path_txt + name + '.txt'
        ap= name.split('_')
        page = ap[len(ap)-1]

        # se non appartengono allo stesso doc salvo precedenti
        print(get_name(dir_list[ind-1])[:-2], name[:-2])
        if ind > 0 and get_name(dir_list[ind-1])[:-2] != name[:-2]:
            save_json = path_json_save + get_name(dir_list[ind-1])[:-2] + '.json' 
            with open(save_json, 'w') as f:
                json.dump(data, f, indent=4)
            data = []

        detect = images_detect + name + type_img
        save_image = path_save + d   #''
          
        new_bb, new_labels = get_bb_merge(path_image, txt, '', detect)
        sorted_boxes = sort_bounding_boxes(new_bb, new_labels)
        data = create_json(data, sorted_boxes, page)

        ind = ind+1

def get_name(d):
    name = '.'.join(d.split('.')[0:len(d.split('.'))-1])
    return name


def sort_data(bb, labels, index):
    combined = list(zip(bb, labels))
    sorted_combined = sorted(combined, key=lambda x: x[0][index])
    sorted_bb, sorted_labels = zip(*sorted_combined)
    return sorted_bb, sorted_labels


# TODO da mettere in utils
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


# TODO da mettere in utils
def create_folder(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

# Funzione per determinare la posizione orizzontale di una bounding box
def get_position(x0):
    if x0 <  250:
        return "sx"
    else:
        return "dx"

# Funzione per ordinare le bounding box alla fine!
def sort_bounding_boxes(boxes, labels):
    box_label_pairs = list(zip(boxes, labels))
    # Ordina prima le bounding box e poi le labels associate
    sorted_box_label_pairs = sorted(box_label_pairs, key=lambda pair: (1 if get_position(pair[0][0]) == "sx" else  2, pair[0][1]))
    # Ricostruisce l'ordine originale delle labels
    return sorted_box_label_pairs

alab = ["sec1", "sec2","sec3","fstline","para","equ","tab","fig","meta","other"] # New 9
#alab = ["title", "sec1", "sec2","sec3","fstline","equ","tab","fig","other"]
def create_json(data, sorted_boxes, page):
    #text
    for i in range(len(sorted_boxes)):
        dict = {"box": sorted_boxes[i][0], 
                "class": alab[sorted_boxes[i][1]],
                "page": page}
       
        data.append(dict)    
    return data
        
    ''
    # devo salvare le bb trovate con la classe
    # per vederlo plotto 2 imm .con id ..
    # è come marge ma assegno solo id 

    #qui salvo solo ( no confronto con gt)
    #with open(json_gt, errors="ignore") as json_file:
    #    data_gt = json.load(json_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
    #folder = 'exp_test/'
    #folder = 'C:/Users/ninad/Desktop/Ok_test_exp2_stat_21_2109.00464_vis/'
    #folder = 'C:/Users/ninad/Desktop/ACL_P10-1160_exp2/'

    folder = 'exp_yolo_9/'#'C:/Users/ninad/Desktop/test_Exp_S/' #1501.04826/'
    merge_all_image(folder, '.png')
