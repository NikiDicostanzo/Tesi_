import os 
import json
import numpy as np

import torch as th
import dgl
import dgl.data
from sklearn.preprocessing import LabelEncoder

from create_graphGT_3lab import calculate_relative_coordinates, get_area, normalize_bounding_box


device = th.device("cuda" if th.cuda.is_available() else "cpu")
class_name = ['title', 'sec', 'meta', 'caption' , 'para', 'note', 'equ', 'tab', 'alg', 'fig', 'page']

def get_info_json(data):
    bounding_boxes = [item['box'] for item in data] # salvo tutte le bb delle miei pagine
    page = [item['page'] for item in data]  
    size = [item['size'] for item in data] 
    text = [item['text'] for item in data]
    type = [item['type'] for item in data]
    style = [item['style'] for item in data]
    font = [item['font'] for item in data]

    labels = [item['class'] for item in data]
    #le = LabelEncoder()
    #encoded_labels = le.fit_transform(labels)

    desired_encoding = {'title': 0, 'sec': 1, 'meta': 2, 'caption': 3, 'para': 4, 'note': 5, 'equ': 6, 'tab': 7, 'alg': 8, 'fig': 9,'page': 10}
    encoded_labels = [desired_encoding[label] for label in labels]
    return bounding_boxes, page, text, size, type, style, font, encoded_labels

def get_all_graphs(folder, array_features):
    path_json = folder #'HRDS/' + type +'/'
    list_j = os.listdir(path_json)
    all_graph = []
    pages = []
    centr =[]
    texts = []
    c = 0
    print(len(list_j))
    for j in list_j:
        print(c)
        json = path_json + j
        g, page, centroid, text = get_graph(json, array_features)
       # print(g)
        all_graph.append(g)
        pages.append(page)
        centr.append(centroid)
        texts.append(text)
        c = c+1
    print('QUI1_ fine ciclo')
    return all_graph#, np.concatenate(pages, axis=0), np.concatenate(centr, axis=0), np.concatenate(texts, axis=0)
   
def get_edge_node(bb):
    # gli elementi sono in ordine li collego semplicemente (?)
    i_node = []
    j_node = []
    for i in range(len(bb)):
        j = 1
        while i + j < len(bb) and j < 3:
            #if  page[i] == page[j]:
            i_node.append(i)
            j_node.append(i + j)
            j = j + 1
    return i_node, j_node


def get_graph(json_file, array_features):
 
    #json_file = folder + 'ACL_2020.acl-main.99.json'
    with open(json_file) as f:
        data = json.load(f)
        bb, page, text, size, type, style, font, labels = get_info_json(data)
    
        n_bb = [(normalize_bounding_box(box, 596, 842)) for box in bb ]
        centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in n_bb ]
      #  dim = [((box[2] + box[0]), (box[3] - box[1])) for box in n_bb ]
      #  #distances = cdist(centroids, centroids)  # Matrice distanza con ogni punto

        i, j = get_edge_node(bb)

        # Graph
        g = dgl.graph((i, j))

          # PREDIRE
        g.ndata['labels'] = th.tensor(labels) 
       # print('Labels')

        #Node Features
        g.ndata['centroids'] = th.tensor(centroids)
        g.ndata['bb'] = th.tensor(n_bb)
        g.ndata['size'] = th.tensor(size)
       # num_page = page[len(page)-1]
            
        if 'rel' in array_features:
          kr = 15
        # Calcola le coordinate relative per ogni nodo
          relative_coordinates = calculate_relative_coordinates(n_bb, kr)
          g.ndata['relative_coordinates'] = th.tensor(relative_coordinates)
       #   print('Rel')

        areas_array, widths, heights = get_area(n_bb)
        if 'area' in array_features:
            g.ndata['area'] = th.tensor(areas_array)
         
         #   print('Area')

        if 'w' in array_features:
            g.ndata['widths'] = th.tensor(widths)
         #   print('Width')

        if 'h' in array_features:
            g.ndata['heights'] = th.tensor(heights)
          #  print('Heights')

    return g, page, centroids, text
        
if __name__ == '__main__':
   # dir = 'acl_anthology_pdfs/'
   # pdf = '2022.naacl-main.92.pdf'#2023.acl-long.150.pdf'# #solo per ridimensionare imm.
    save_path = 'plot_bb_parse/'

   # if not os.path.exists(save_path):
   #     os.makedirs(save_path)
    folder = 'yolo_hrds_4_gt_test/check_json_label/'
    array_features = ['bb', 'cent', 'area', 'w', 'h', 'rel']
    get_all_graphs(folder, array_features)
    # g, page, centroids, text = get_graph(folder, array_features)
    # print(g)


""" 
    Graph(num_nodes=900, num_edges=1797,
      ndata_schemes={'labels': Scheme(shape=(), dtype=torch.int64), 'centroids': Scheme(shape=(2,), dtype=torch.float32), 'bb': Scheme(shape=(4,), dtype=torch.float32), 'relative_coordinates': Scheme(shape=(6, 4), dtype=torch.float64), 'area': Scheme(shape=(), dtype=torch.float64), 'widths': Scheme(shape=(), dtype=torch.float32), 'heights': Scheme(shape=(), dtype=torch.float32)}
      edata_schemes={}) 
"""