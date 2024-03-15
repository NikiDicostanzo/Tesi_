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

from plot_edge_multipage_YOLO import get_nodes
from create_graphGT_3lab import add_edge, calculate_relative_coordinates, normalize_bounding_box, processing_lab, title_condition

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


def condition_edge(bounding_boxes, labels_yolo, index_i, index_j, distances):
    return abs(distances) < 15 and (labels_yolo[index_i]== labels_yolo[index_j] \
                                        or labels_yolo[index_i] == 'fstline') \
                    or (abs(bounding_boxes[index_j][0] - bounding_boxes[index_i][0])>200 and labels_yolo[index_i]== labels_yolo[index_j] and labels_yolo[index_i]!= 'meta')
 
def plot_bb(box, labels_yolo, draw):
    index = 0
    for bb in box:
        name_class = labels_yolo[index]
        class_lab = get_color(name_class)
        draw.rectangle(bb, outline = class_lab)
        index = index +1 

def get_color(name_class):
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
    return class_lab

def disty_vert(box1, box2):
    distY = box2[1] - box1[3] # trovo distanza con quelle sotto 
    # vertice alto -> box2  (y0)
    # vertice basso -> box1 (y1)
    return distY

def distx_oriz(box1, box2):
    distX = (box2[2] - box1[2])  # trovo distanza con quelle a dx 
    return distX

def get_info_json(data):
    bounding_boxes = [item['box'] for item in data] # salvo tutte le bb delle miei pagine
    page = [int(item['page']) for item in data] 
    labels = [item['class'] for item in data]
    return bounding_boxes, page,labels

def get_name(path_image, name, page, u):
    name_img = path_image + name + '_' + str(page[u]) +'.png'
    if os.path.exists(name_img):
        image = Image.open(name_img)
        wid = image.width 
    else:
        image = None
        wid = 0
        #draw = D.Draw(image)
    return image, wid

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    draw = ImageDraw.Draw(dst)
    return dst, draw

def plot_edge(name, page, new_cent, u, v, image1, image2, lab, index):
    path_save_conc = 'zexp_yolo_9_hrdh/savepred_2classi/' + name + '_' + str(page[u]) +'_'+ str(page[v]) + '.png'# +'_'+ str(index) + '.png'
    con_img, con_draw = get_concat_h(image1, image2)#.save(path_save_conc)
    index = 0
    for cu, cv in new_cent:
        if lab[index] == 1:
            color = 'blue'
            wid = 2
        else:
            color = 'red'
            wid = 1
        
        con_draw.line([tuple(cu), tuple(cv)], fill=color, width=wid)
        index = index + 1
        
    con_img.save(path_save_conc)# cambiare pagina

def plot_box_yolo(draw, get_color, get_name, plot_edge, path_image, new_cent, num_page, lab, path_new_im, bounding_boxes, page, labels_yolo, centroids, labels, i, j, name, image):
    set_page = set(page)
    for b in range(len(bounding_boxes)):
        p = page[b]
        #print(p, len(bounding_boxes))
        color = get_color(labels_yolo[b])
        if b>0 and p != page[b-1]:# or p == len(bounding_boxes)-1: #cambia pagina #salvo
            save_im = path_new_im + name + '_' + str(page[b-1]) + '.png'
            image.save(save_im)
            image, _ = get_name(path_image, name, page, b)
            draw = ImageDraw.Draw(image)
        
        draw.rectangle(bounding_boxes[b], outline = color) 
        if b == len(bounding_boxes)-1:
            save_im = path_new_im + name + '_' + str(page[b]) + '.png'
            image.save(save_im)

def get_graph_yolo():
    path_image = 'zexp_yolo_9_hrdh/images/'
    path_json = 'zexp_yolo_9_hrdh/json_yolo/'
    
    if not os.path.exists('zexp_yolo_9_hrdh/savepred_2classi/'):
         os.makedirs('zexp_yolo_9_hrdh/savepred_2classi')
    array_graph = []
    arr_page =[]
    list_json = os.listdir(path_json)
    for d in list_json: # d = 'ACL_P11-1008.json' 
        new_cent = []
        new_bb = []
       
        num_page = 0
        lab = []
        file_json =  path_json + d
        path_new_im = 'zexp_yolo_9_hrdh/savebox/'
        if not os.path.exists(path_new_im):
            os.makedirs(path_new_im)

        with open(file_json, errors="ignore") as json_file:
            data = json.load(json_file)
            # Tutte le informazioni di tutte le pagine 
            bounding_boxes, page, labels_yolo = get_info_json(data)
            centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in bounding_boxes]

          #  labels, i, j = get_nodes2(bounding_boxes, labels_yolo, page)  
              
            labels, i, j = get_nodes(bounding_boxes, labels_yolo, page)
            name = d.replace('.json', '')

            image, _ = get_name(path_image, name, page, 0)
            save_image = False
            
            if save_image and  image != None: #name in "ACL_2021.acl-long.546" and 
                draw = ImageDraw.Draw(image)
                plot_box_yolo(draw, get_color, get_name, plot_edge, path_image, new_cent, 
                               num_page, lab, path_new_im, bounding_boxes, page, 
                               labels_yolo, centroids, labels, i, j, name, image)


            n_bb = [(normalize_bounding_box(box, 596, 842)) for box in bounding_boxes ]
            n_centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in n_bb ]
            
            #  # Graph
            g = dgl.graph((i, j))   
            
            g.edata['label'] = th.tensor(labels)

            #Node Features
            g.ndata['centroids'] = th.tensor(n_centroids)
            g.ndata['bb'] = th.tensor(bounding_boxes)
            
            relative_coordinates = calculate_relative_coordinates(n_bb)
            g.ndata['relative_coordinates'] = th.tensor(relative_coordinates)

            encoded_labels = processing_lab(labels_yolo)

          #  print(len(encoded_labels), len(n_centroids),  len(bounding_boxes))
            g.ndata['labels'] = th.tensor(encoded_labels) 

            array_graph.append(g)
            arr_page.append(page)
    return array_graph, np.concatenate(arr_page, axis=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
        
    #'exp_yolo_9/json_yolo/ACL_P11-1008_5.json' #ACL_2020.acl-main.99_0.json'
  
    array_graph= get_graph_yolo()
   # print(array_graph)
    bg_train = dgl.batch(array_graph) # num_nodes=725391, num_edges=811734,
    #bg_train = bg_train.int().to(device)
    print(bg_train)
#
             