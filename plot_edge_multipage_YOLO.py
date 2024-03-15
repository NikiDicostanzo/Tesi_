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

from create_graphGT_3lab import calculate_relative_coordinates, normalize_bounding_box, processing_lab

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
    count_element_page = 0
    second_page  = False
    for index_i in range(len(bounding_boxes)):
        k = 0
        #index_j = index_i + k
        index_j = index_i
        # voglio solo i k + vicini
        count_edge = 0
        blu_plot = True
        if index_i > 0 and page[index_i] == page[index_i-1] :
            count_element_page = count_element_page + 1 # conto elementi pagina e azzero quando salvo
        else:
            count_element_page = 1
            blu_plot_new = True
        while k < 10 and index_j-k >= 0 : #index_j < len(bounding_boxes):
            distances =  min_disty_vert(bounding_boxes[index_j-k], bounding_boxes[index_i])
            if page[index_j-k] == page[index_i]: 
                if blu_plot and condition_edge(bounding_boxes, labels_yolo, index_j-k, index_i, distances):
                  
                    j_node.append(index_i)
                    i_node.append(index_j-k)
                    labels.append(1)
                    count_edge = count_edge+1 
                    blu_plot = False  
                    #break
                elif count_edge<2: 
                    j_node.append(index_i)
                    i_node.append(index_j-k) 
                    labels.append(0)
                    count_edge = count_edge+1
                    
                #if labels_yolo == 'meta'
            elif labels_yolo[index_j-k] != 'meta':#if labels_yolo[index_j] != 'meta' and labels_yolo[index_j] != 'tab':
                # se ho : meta, tab, fig 
                
             #   print(labels_yolo[index_j-k], labels_yolo[index_i])    
                        #    if labels_yolo[index_j-k] == labels_yolo[index_i]:
                if blu_plot_new and condition_edge_page(count_element_page, labels_yolo, index_j-k, index_i, page):
                    j_node.append(index_i)
                    i_node.append(index_j-k)
                    labels.append(1)
                    count_edge = count_edge+1 
                    blu_plot_new = False  
                    #break
                elif count_edge<2: 
                    j_node.append(index_i)
                    i_node.append(index_j-k) 
                    labels.append(0)
                    count_edge = count_edge+1
                        
                       
            k=k+1
           # index_j = index_i + k
    
    j = th.tensor(j_node)
    i = th.tensor(i_node)
    return labels, i, j

def condition_edge_page(count_element_page, labels_yolo, index_i, index_j, page):
    if labels_yolo[index_j] in ['para', 'fstline'] and labels_yolo[index_i] in ['para', 'fstline']:
        return True
    return False

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

def min_disty_vert(box1, box2):
    distY = box2[1] - box1[3] # trovo distanza con quelle sotto 
    # vertice alto -> box2  (y0)
    # vertice basso -> box1 (y1)
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

def plot_edge_yolo(draw, get_color, get_name, plot_edge, path_image, new_cent, num_page, lab, path_new_im, bounding_boxes, page, labels_yolo, centroids, labels, i, j, name, image):
    for b in range(len(bounding_boxes)):
        p = page[b]
        color = get_color(labels_yolo[b])
        if b>0 and p != page[b-1]: #cambia pagina #salvo
            save_im = path_new_im + name + '_' + str(page[b-1]) + '.png'
            image.save(save_im)
            image, _ = get_name(path_image, name, page, b)
            draw = ImageDraw.Draw(image)
        elif b == len(bounding_boxes)-1:
            save_im = path_new_im + name + '_' + str(page[b]) + '.png'
            image.save(save_im)
        draw.rectangle(bounding_boxes[b], outline = color) 
                
    save = True
    for index in range(len(j)):
        u = i[index]
        v = j[index]
                #  print(page[u], page[v])
        if page[u] == page[v]:  # Se la pagina è la stessa, aggiungi i centroidi e le bb
                new_cent.append([centroids[u], centroids[v]])
                lab.append(labels[index])
                save = True
        elif save and page[u]!= page[v]:
                        # Devi spostare i centroidi di v e aggiungere tutte le altre informazioni
            k = index + 1
            if k < len(j):
                u1 = i[k]
                v1 = j[k]
            image1, width = get_name(path_new_im, name, page, u)
            image2, _ = get_name(path_new_im, name, page, v)

                        # Plotto gli edge del 2 documento a dx
            while k < len(j) and int(page[u]) < int(page[v1]) <= int(page[u]) + 1:
                if page[u1] != page[v1]:
                    new_cent.append(tuple([[centroids[u1][0], centroids[u1][1]], [centroids[v1][0] +  width, centroids[v1][1]]]))
                else:
                    new_cent.append(tuple([[centroids[u1][0] +  width, centroids[u1][1]], [centroids[v1][0] +  width, centroids[v1][1]]]))
                lab.append(labels[k])
                k = k +  1
                if k < len(j):
                    u1 = i[k]
                    v1 = j[k]
                    #  print(len(lab), len(new_cent))
            plot_edge(name, page, new_cent, u, v, image1, image2, lab, index)
                    #  print(name)
            save = False
            new_cent = []
            lab = []
            num_page =num_page + 1

def get_graph():
    path_image = 'zexp_yolo_9_hrdh/images/'
    path_json = 'zexp_yolo_9_hrdh/json_yolo/'
    
    if not os.path.exists('zexp_yolo_9_hrdh/savepred_2classi/'):
         os.makedirs('zexp_yolo_9_hrdh/savepred_2classi')
    array_graph = []
    list_json = os.listdir(path_json)
    for d in list_json: # d = 'ACL_P11-1008.json' 
        new_cent = []
        new_bb = []
        num_page = 0
        lab = []
      #  print(d)
        file_json =  path_json + d
        path_new_im = 'zexp_yolo_9_hrdh/savebox/'
        if not os.path.exists(path_new_im):
            os.makedirs(path_new_im)

        with open(file_json, errors="ignore") as json_file:
            data = json.load(json_file)
            # Tutte le informazioni di tutte le pagine 
            bounding_boxes, page, labels_yolo = get_info_json(data)
            centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in bounding_boxes]

            labels, i, j = get_nodes(bounding_boxes, labels_yolo, page)      
            name = d.replace('.json', '')

            image, _ = get_name(path_image, name, page, 0)
            
            # if image != None:
            #     draw = ImageDraw.Draw(image)
            #     plot_edge_yolo(draw, get_color, get_name, plot_edge, path_image, new_cent, 
            #                    num_page, lab, path_new_im, bounding_boxes, page, 
            #                    labels_yolo, centroids, labels, i, j, name, image)


            n_bb = [(normalize_bounding_box(box, 596, 842)) for box in bounding_boxes ]
            n_centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in n_bb ]
            
             # Graph
            g = dgl.graph((i, j))
            
            g.edata['label'] = th.tensor(labels)

            #Node Features
            g.ndata['centroids'] = th.tensor(n_centroids)
            g.ndata['bb'] = th.tensor(bounding_boxes)
            
            # relative_coordinates = calculate_relative_coordinates(n_bb)
            # g.ndata['relative_coordinates'] = th.tensor(relative_coordinates)

            # encoded_labels = processing_lab(labels_yolo)

            # print(len(encoded_labels), len(n_centroids),  len(bounding_boxes))
            # g.ndata['labels'] = th.tensor(encoded_labels) 

            array_graph.append(g)
    return array_graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
        
    #'exp_yolo_9/json_yolo/ACL_P11-1008_5.json' #ACL_2020.acl-main.99_0.json'
  
    array_graph= get_graph()
   # print(array_graph)
    bg_train = dgl.batch(array_graph) # num_nodes=725391, num_edges=811734,
    #bg_train = bg_train.int().to(device)
    print(bg_train)
#
             