import argparse
import json

import dgl
import numpy as np
import torch as th
import os
from dgl.nn import GATv2Conv
from scipy.spatial.distance import cdist

import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw

#from torch_geometric.data import Data

# ordinare i dati!!!! 
'''
capire dove si trovano -> calolare la sua lunghezza x! 
    - se è maggiore di 300 : si trova non nelle colonne 
    - se e minore :
        vedere la x0 se:
        - minore di 200 (?) allora sta a sx
        - altrimenti a destra
    -->voglio che siano in ordine in base alla Y:
        però se si trova a dx quindi nella colonna di destra vengono dopo quelli di sx
'''

def get_nodes(bounding_boxes, labels_yolo, page):
    j_node = [] 
    i_node= []
    labels=[] # per ogni arco metto una labels 

    # considerare i k piu vicini !! 
    # Dati sono in ordine di lettura

    for index_i in range(len(centroids)):
        k = 1
        index_j = index_i + k
        #print(len(i_node), len(j_node))
        # voglio solo i k + vicini
       # for index_j in range(index_i+1, len(centroids)):
        while k < 3 and index_j < len(centroids):
            j_node.append(index_j)
            i_node.append(index_i)
            distances =  min_disty_vert(bounding_boxes[index_i], bounding_boxes[index_j])
            # fstline ! e caso cambio paragrafo
            print(distances, labels_yolo[index_i], labels_yolo[index_j])
            if 0 < distances < 30 and (labels_yolo[index_i]== labels_yolo[index_j] \
                                       or labels_yolo[index_i] == 'fstline') \
                or (bounding_boxes[index_j][0] - bounding_boxes[index_i][0]>200 and labels_yolo[index_i]== labels_yolo[index_j]):
                labels.append(1)
            else :  # if 20<=distances[index_i][index_j]<100:
                labels.append(0)
            #else:
            #    labels.append(2)
            
            k=k+1
            index_j = index_i + k
    
    j = th.tensor(j_node)
    i = th.tensor(i_node)
    return labels,j,i
 
def plot_bb(box, labels_yolo, draw):
    index = 0
    for bb in box:
        name_class = labels_yolo[index]
        if name_class == 'sec1':
            class_lab = 'red'
        elif name_class == 'sec2':
            class_lab = 'red'
        elif name_class == 'sec3':
            class_lab = 'red'
        elif name_class == 'fstline':
            class_lab = 'orange'
        elif name_class == 'para':
            class_lab = 'blue'
        elif name_class == 'equ':
            class_lab = 'cyan'
        elif name_class == 'tab':
            class_lab = 'yellow'
        elif name_class == 'fig':
            class_lab = 'magenta'
        elif name_class == 'is_meta':
            class_lab = 'green' 
        else:
            class_lab = 'black'
        draw.rectangle(bb, outline = class_lab)
        index = index +1 

def min_disty_vert(box1, box2):
    distY = box2[3] - box1[3] # trovo distanza con quelle sotto 
    return distY

def min_distx_oriz(box1, box2):
    distX = (box2[2] - box1[2])  # trovo distanza con quelle a dx 
    return distX

def my_graph():
    json_file = 'data_h/json/ACL_2020.acl-main.99.json'
    path_save = 'data_h/save/'
    folder = 'data_h/image/'
    list_image = os.listdir(folder)
   # print(list_image)
    #for page in range(len(list_image)):
    page = 0
    path_image = folder + list_image[page]
    image_save = path_save + list_image[page]
    #get_near(json_file, page, path_image, image_save)

#def edge_more_page():


def get_info_json(data):
    bounding_boxes = [item['box'] for item in data] # salvo tutte le bb delle miei pagine
    page = [item['page'] for item in data] 
    labels = [item['class'] for item in data]
 
    return bounding_boxes, page,labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
        
    #'exp_yolo_9/json_yolo/ACL_P11-1008_5.json' #ACL_2020.acl-main.99_0.json'
    file_json = 'ACL_P11-1008.json' 
    with open(file_json, errors="ignore") as json_file:
        data = json.load(json_file)
        bounding_boxes, page, labels_yolo = get_info_json(data)
        print(labels_yolo)
        centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in bounding_boxes ]

        labels,j,i = get_nodes(bounding_boxes, labels_yolo, page)

        name_img = 'exp_yolo_9/images/ACL_P11-1008_5.png'
        image = Image.open(name_img)
        draw = ImageDraw.Draw(image)
        plot_bb(bounding_boxes, labels_yolo, draw)
        print(labels)
        for k in range(len(labels)):
            node_j = j[k]
            node_i = i[k]
            if labels[k] == 1:
                color = 'blue'
                wid = 2
            else:
                color = 'red'
                wid = 1
            
            draw.line([centroids[node_i], centroids[node_j]], fill=color, width=wid)
        image.save('ACL_P11-1008_5.png')


     
""" 
    #itero direttamente sui json : 
    json_path = 'HRDS/train/'
    list_json = os.listdir(json_path)
    for j in list_json:
        json = json_path + j  """

