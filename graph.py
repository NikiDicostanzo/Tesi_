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

from PIL import Image, ImageDraw

#from torch_geometric.data import Data

def get_nodes(centroids, distances):
    j_node = [] 
    i_node= []
    labels=[] # per ogni arco metto una labels 
    for index_i in range(len(centroids)):
        #index_i = 0
        #print(len(i_node), len(j_node))
        # voglio solo i k + vicini
        for index_j in range(index_i+1, len(centroids)):
            j_node.append(index_j)
            i_node.append(index_i)
            if distances[index_i][index_j]<20:
                labels.append(0)
            else :  # if 20<=distances[index_i][index_j]<100:
                labels.append(1)
            #else:
            #    labels.append(2)
       #     k=k+1
    
    j = th.tensor(j_node)
    i = th.tensor(i_node)
    return labels,j,i

def get_relation(data):
    relation_dict = {'connect': 0, 'meta': 1, 'contain': 2, 'equality': 3}
    relations = [relation_dict[item['relation']] for item in data]
    print(relation_dict, set(relations))
    return relations

def plot_data(draw, g):
    G = dgl.to_networkx(g)
    print((G.edges()))
    i = 0
    centroids = [tuple(row) for row in g.ndata['centroids'].tolist()]
    for u, v in G.edges():#g.edges():
        if g.edata['label'][i] == 0:
            #print('qui', type(centroids[u]))
            draw.line([centroids[u], centroids[v]], fill='blue', width=2)
        #elif labels[i] == 1:
        #    draw.line([centroids[u], centroids[v]], fill='red', width=1)
        i=i+1
    for centroid in centroids:#G.nodes():
        draw.ellipse([centroid[0] - 2, centroid[1] - 2, centroid[0] + 2, centroid[1] + 2], outline='red')
        
def plot_bb(box, draw):
    for bb in box:
        draw.rectangle(bb, outline = 'black') 

def min_dist_x(box1, box2):
    distX = abs(box1[1] - box2[2]) # mi serve solo per trovare un punto sulla x
    distY = box2[3] - box1[3] # trovo distanza con quelle sotto 

    return distX, distY

def min_dist_y(box1, box2):
    distY = abs(box2[3] - box1[3]) # mi serve solo per trovare un punto sulla y
    distX = (box2[2] - box1[2])  # trovo distanza con quelle a dx 
    return distX, distY

# per ogni bb ho + punti ? -> k punti piu vicini-> tanti centroidi quanti archi o per 2

def get_near(json_file, page, path_image, path_save):
     with open(json_file) as f:
        data = json.load(f)
        bounding_boxes = [item['box'] for item in data if item['page'] == page] #Per una pagina!!
        array_edges=[]
        
        for i in range(len(bounding_boxes)):
            #node = {'point':[], 'edges':[]}
            #edges=[]
            k = i + 1
            #devo confrontare le box prima verticalmente e poi distanza orizzontale
            #per ogni box ho piu distanze, salvo distanze e coppia di nodi (box) ->
            # {nodo: i, centri : [[1,2], [13,1], [1,44], [17,8]], archi:[4,5,17,19]}
            # ho un dizionario ad esempio nodo i = 3 quindi nella posizione 3 ho 2 array
            while k<15: #len(bounding_boxes): # considero prima verticali box (?)  
                distX, distY = min_dist_x(bounding_boxes[i], bounding_boxes[k])
                print(data[i]['text'], ' - ', distY, ' - ', data[k]['text'])#distY)
                if distY < 18:
                    array_edges.append([i,k])
                    #node['edges'].append(k)
                   # node['point'].append([]) # se ho arco questo posso ricalcolarlo, (capire come distinguire con quelle a dx, laterali)

                #salvo distanze minore di 15 (y)
                k= k+1
           #array_edges.append(edges) # 
        
        point = get_point(array_edges, bounding_boxes)

        img = Image.open(path_image)
        draw = ImageDraw.Draw(img)
        plot_data_2(draw, point)
        img.save(path_save)

def get_point(array_edges, box):
    point = []
    for edge in array_edges:
        # i[0] , i[1] # indice box su , box giu
        i =edge[0] 
        j = edge[1]

        x = get_intersection(box[i], box[j])
        if x == None:
            x = box[i][0] # se non sono allineate

        point_i = [x, box[i][3]] #y1
        point_j = [x, box[j][1]] #y0
        point.append([point_i, point_j])
    return point


def plot_data_2(draw, point):
    
    for p in point:
       # if g.edata['label'][i] == 0:
        #print(type(p[0]))
        draw.line([tuple(p[0]), tuple(p[1])], fill='blue', width=2)
        #elif labels[i] == 1:
        #    draw.line([centroids[u], centroids[v]], fill='red', width=1)
        for i in p:#G.nodes():
            draw.ellipse([i[0] - 2, i[1] - 2, i[0] + 2, i[1] + 2], outline='red')

def get_intersection(box1, box2):
        l = max(box1[0], box2[0])
        r = min(box1[1], box2[1])
        if l <= r:
            return (l + r)/2 #[l, r]
        else:
            return None



# per ogni nodo ho una tupla ? 

def get_graph(json_file, page, path_image, path_save):
    with open(json_file) as f:
        data = json.load(f)
        bounding_boxes = [item['box'] for item in data if item['page'] == page] #len = 92 (902 tot)

        # {'connect', 'meta', 'contain', 'equality'}
        #[x0,y0,x1,x1]
        centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in bounding_boxes ]
        distances = cdist(centroids, centroids)  # Matrice distanza con ogni punto
        print((distances)) 

        labels, j, i = get_nodes(centroids, distances)
        print(len(i), len(j))

        # Graph
        g = dgl.graph((i, j))
        print('LABELS', len(labels), g.number_of_edges())
        g.edata['label'] = th.tensor(labels)
        print('ui', len(g.edata['label']))
        g.ndata['centroids'] = th.tensor(centroids)
        g.ndata['bb'] = th.tensor(bounding_boxes)

        #printg)
        #print(g.edata)

        relations = get_relation(data)
        #plot(path_image, path_save, bounding_boxes, g)
        g = dgl.add_self_loop(g)
        print(g.number_of_nodes(), labels)
        return g


def gat(g):
    gat_layer = GATv2Conv(in_feats=5, out_feats=2, num_heads=3)
    g = dgl.add_self_loop(g)
    num_nodes = g.number_of_nodes(); 
    print('num_nodes: ', num_nodes)
    h = th.randn(num_nodes, 5) # Assuming you have 10 features for each node
    res = gat_layer(g, h)
    print(res)

def plot(path_image, path_save, bounding_boxes, g):
    img = Image.open(path_image)
    draw = ImageDraw.Draw(img)
    plot_bb(bounding_boxes, draw)
    plot_data(draw, g)
   # img.save(path_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
    
    json_file = 'data_h/json/ACL_2020.acl-main.99.json'
    path_save = 'data_h/save/'
    folder = 'data_h/image/'
    list_image = os.listdir(folder)
   # print(list_image)
    for page in range(len(list_image)):
        path_image = folder + list_image[page]
        image_save = path_save + list_image[page]
        #get_near(json_file, page, path_image, image_save)
        g = get_graph(json_file, page, path_image, image_save)
        gat(g)