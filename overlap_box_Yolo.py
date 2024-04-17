import csv
import json
import argparse
import os
from PIL import Image, ImageDraw as D
import numpy as np
from plot_edge_multipage_YOLO import get_color

def get_name(d):
    name = '.'.join(d.split('.')[0:len(d.split('.'))-1])
    return name

def sort_data(bb, labels, confidence, index):
    combined = list(zip(bb, labels, confidence))
    sorted_combined = sorted(combined, key=lambda x: x[0][index])
    sorted_bb, sorted_labels, sorted_confidence = zip(*sorted_combined)
    return sorted_bb, sorted_labels, sorted_confidence


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

#TODO #################
alab = ["sec1", "sec2","sec3","fstline","para","equ","tab","fig","meta","other"] # New 9
#alab = ["sec","para","fig","meta","cap"] # New 9

#alab = ["title", "sec1", "sec2","sec3","fstline","equ","tab","fig","other"]
def create_json(data, sorted_boxes, page):
    #text
    for i in range(len(sorted_boxes)):
        dict = {"box": sorted_boxes[i][0], 
                "class": alab[sorted_boxes[i][1]],
                "page": page}
       
        data.append(dict)    
    return data

# Covert w, h, and center to bb
def get_bbox_coords(c0, c1, h, w, width, height):
    x_min = (c0 - w / 2) * width
    y_min = (c1 - h / 2) * height
    x_max = (c0 + w / 2) * width
    y_max = (c1 + h / 2) * height
    return (x_min, y_min, x_max, y_max)

# Get bb from labels of detection
def get_bb(file_path, image):
    bb = []
    line = []
    type = []
    confidence = []
    
    file = open(file_path)
    
    width, height = image.size 
    for i in file:
        line = i.split("\n")[0].split(" ")
        
        c0 = float(line[1])
        c1 = float(line[2])
        w = float(line[3])
        h = float(line[4])
        get_bbox_coords(c0, c1, h, w, width, height)
        confidence.append(float(line[5]))
        bb.append(get_bbox_coords(c0, c1, h, w, width, height))
        type.append(int(line[0]))
    return bb, type, confidence

    
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
        color = get_color(alab[labels[k]])
        # #if labels[k] == 0:
        #    color = 'red'
        # else:
        #    color = 'black'
        draw.rectangle(bb[k], outline = color, width = 1) 


def marge_bb(bb, labels, confidence):
  #  path_image = 'dataset/imgs/stat_21_2109.00464_vis/7.png' 
    tmp = []
    tmp_lab = []
    remove = []
    remove_lab = []
    new_labes = []
    new_bb = []
    new_conf = []

    k = 0
    while k < (len(bb)): 
        j = k + 1 # parte da quello successivo
        x0 = bb[k][0]
        y0 = bb[k][1]
        x1 = bb[k][2]
        y1 = bb[k][3]
        marge_box = False
        my_lab = []
        while j < (len(bb)): 
            iou = check_overlap([x0,y0,x1,y1], bb[j]) 
           
            
            if iou > 0.0 and abs(bb[k][0]-bb[j][0]) < 150:# and (labels[k] == labels[j] or labels[k] == 3):    #fstline = 3
              #  print(bb[j][0]-x0)
                           # si sovrappongono 
               #print(labels[k], labels[j], '/n')
                # new bounding box 
                x0 = min(bb[j][0], x0)
                y0 = min(bb[j][1], y0)
                x1 = max(bb[j][2], x1) 
                y1 = max(bb[j][3], y1)

                remove.append(j) # rimuovo box
                # tolgo indice che a conf <
                if confidence[k] > confidence[j]:
                    my_lab = labels[k]
                else:
                    my_lab = labels[j]
                #####TODO Stessa classe
                marge_box = True                    
            j +=1
            # elfi # aggiugnere se si sopvrappongono e classi diverse di prendere quella con confidenza >
        if marge_box : 
            #if [x0,y0,x1,y1] not in tmp:
            tmp.append([x0,y0,x1,y1])  
            marge_box = False
            tmp_lab.append(my_lab)
        else:
            tmp.append(bb[k])
            tmp_lab.append(labels[k])
        k +=1 
        
    remove = sorted(list(set(remove)))
    for i in range(len(tmp)):
         if i not in remove:
            new_bb.append(tmp[i])
            new_labes.append(tmp_lab[i])
            new_conf.append(confidence[i])
   # print(len(new_labes), len(new_bb))
    return new_bb, new_labes, new_conf

# Merge su una immagine
def get_bb_merge(path_image, txt, path_save, detect):
    image = Image.open(path_image)
    bb, labels, confidence = get_bb(txt,image)

    sorted_bb_y, sorted_labels_y, sorted_confidence_y  = sort_data(bb, labels, confidence, 1)
    bb_y, labels_y, confidence_y = marge_bb(list(sorted_bb_y), list(sorted_labels_y), list(sorted_confidence_y)) #y

    sorted_bb_x, sorted_labels_x, sorted_confidence_x = sort_data(bb_y, labels_y, confidence_y, 0)
    new_bb, new_labels, _ = marge_bb(list(sorted_bb_x), list(sorted_labels_x), list(sorted_confidence_x)) #x
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
import natsort
# Merge su tutte le immagini
def merge_all_image(folder, type_img):
    # txt e img devono avere stesso nome!
    path_images = folder + 'images/'#'zexp_yolo_9_hrdh/images/'
     #folder + 'images/' '../dataset_yolo_hrds/test/images/'#
    path_txt = folder + 'labels/'
    images_detect = folder + 'detect/'

    dir_list = os.listdir(path_images) 
    sorted_dir_list = natsort.natsorted(dir_list)   # print(dir_list)
    path_save = folder + 'merge/'
    create_folder(path_save)

    path_json_save = folder + 'json_yolo/'
    create_folder(path_json_save)
    
    ind = 0
    data = []

    for d in sorted_dir_list: # ciclo su tutte le immagini
        
        path_image = path_images + d 
        name = get_name(d)
        txt = path_txt + name + '.txt'
        ap= name.split('_')
        page = ap[len(ap)-1]

        n = '_'.join(ap[:len(ap)-1])

        # se non appartengono allo stesso doc salvo precedenti nello stesso json
        cn = get_name(sorted_dir_list[ind-1]).split('_')
        check_name = '_'.join(cn[:len(cn)-1])
        print(d)
        if ind > 0 and (check_name != n): # _0  o _10 
            
        #    print(check_name, '|', n)
            
        #    print()
            save_json(sorted_dir_list, path_json_save, ind, data)
            data = []
            
        detect = images_detect + name + type_img
        save_image = path_save + d   #''
        
        # Se le bb si sovrappongono faccio marge
        #if d == 'ACL_P17-1140_9.png': #ACL_2020.acl-main.640_7.png
        #    print('qui')
        if os.path.exists(txt):
            new_bb, new_labels = get_bb_merge(path_image, txt, save_image, detect)
            #Ordino in base alla loro posizione nel doc
            sorted_boxes = sort_bounding_boxes(new_bb, new_labels)
        else:
            new_bb = []
            new_labels = []
            sorted_boxes = []
        data = create_json(data, sorted_boxes, page)

        ind = ind+1

def save_json(dir_list, path_json_save, ind, data):
    name_list = get_name(dir_list[ind-1]).split('_')
    name = '_'.join(name_list[:len(name_list)-1])
    #print(name_list, name)
    save_json = path_json_save + name + '.json' 
    with open(save_json, 'w') as f:
        json.dump(data, f, indent=4)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
    #folder = 'exp_test/'
    #folder = 'C:/Users/ninad/Desktop/Ok_test_exp2_stat_21_2109.00464_vis/'
    #folder = 'C:/Users/ninad/Desktop/ACL_P10-1160_exp2/'

    folder = 'yolo_hrdhs_672_9/'#'C:/Users/ninad/Desktop/test_Exp_S/' #1501.04826/'
    merge_all_image(folder, '.png')
