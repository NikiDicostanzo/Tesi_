# ogni json mi dà 1 grafo ... 

import os
import json 
import dgl
from dgl import save_graphs, load_graphs
import torch as th
from scipy.spatial.distance import cdist
import dgl.data
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from transformers import BertModel, BertTokenizer
device = th.device("cuda" if th.cuda.is_available() else "cpu")
# Caricamento del modello e del tokenizer
model = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with th.no_grad():
        outputs = model(**inputs)
        # Ottieni l'embedding del token [CLS]
        cls_token_embedding = outputs.last_hidden_state[0, 0, :]
    return cls_token_embedding

def get_info_json(data):
    bounding_boxes = [item['box'] for item in data] # salvo tutte le bb delle miei pagine
    page = [item['page'] for item in data] 
    labels = [item['class'] for item in data]
    relation = [item['relation'] for item in data] 
    parent = [item['parent_id'] for item in data] 
    text = [item['text'] for item in data]
    return bounding_boxes, page,relation,parent, labels, text

#creare i labels degli archi 
def get_edge_node(data, bounding_boxes, page, relation, parent):
        labels_edge = []
        array_edges = []
        node_i = []
        node_j = []
        for i in range(len(data)): 
            k = 1
            prova = True
            while k< 10 and i - k >0: #and k<i+2 :
                if page[i] == page[i-k]:
                    if i >0 and relation[i]=='connect' and parent[i] == data[i-k]['line_id']:
                        array_edges.append([i-k,i]) # arco con quello precedente
                        node_i.append(i-k)
                        node_j.append(i)

                        labels_edge.append(1)
                    elif k==1 or bounding_boxes[i-k][0] - bounding_boxes[i][0] >150:
                        array_edges.append([i-k,i]) # arco con quello precedente
                        node_i.append(i-k)
                        node_j.append(i)

                        labels_edge.append(0)
                else:
                    
                    if i >0  and relation[i-k] !='meta':# and k>2:
                        
                        if relation[i]=='connect' and parent[i] == data[i-k]['line_id']:
                            array_edges.append([i-k,i]) # arco con quello precedente
                            node_i.append(i-k)
                            node_j.append(i)

                            labels_edge.append(1)
                           # prova = True
                       #     print('i-k', i-k, data[i-k]['text'],'|','i', i, data[i]['text'] ,'|')
                       #     print(relation[i-k], relation[i], 1, '/n')
                          #  prova = False
                        elif prova == True: #salvo solo 1 rosso
                            array_edges.append([i-k,i])
                            node_i.append(i-k)
                            node_j.append(i)

                            labels_edge.append(0)
                     #       print('i',i, 'i-k',i-k, 'k', k, '|',data[i-k]['text'],'|',data[i]['text'] ,'|') 
                     #       print(relation[i-k], relation[i], 0, '/n')
                            prova = False
                k = k + 1
            # array_edges
            
        return node_i, node_j, labels_edge

import numpy as np


def calculate_relative_coordinates(bb, k=10):
    # Assumendo che g.ndata['bb'] contenga le bounding boxes normalizzate
    relative_coordinates = []
    for i in range(len(bb)):
       
        current_bb = np.array(bb[i])
        if i + k < len(bb):
            next_bbs = np.array(bb[i+1:i+k+1])
        else:
            next_bbs = np.zeros(k) #np.array([]) #TODO rivedere
        # Calcola la differenza relativa tra la bounding box corrente e quelle dei k vicini successivi
       # relative_diffs = [np.abs(current_bb - next_bb) for next_bb in next_bbs]
        relative_diffs = [(next_bb - current_bb) for next_bb in next_bbs]

        relative_coordinates.append(relative_diffs)
    return np.array(relative_coordinates)

def normalize_bounding_box(box, image_width, image_height):
    # Normalizzazione diretta delle coordinate x e y
    normalized_x0 = box[0] / image_width
    normalized_y0 = box[1] / image_height
    normalized_x1 = box[2] / image_width
    normalized_y1 = box[3] / image_height
    
    return normalized_x0, normalized_y0, normalized_x1, normalized_y1
def get_all_bb_rotolone(page, box):
    somma_larghezza = 0
    new_box = []
    for i in range(len(box)):
        somma_larghezza = page[i] * 560
        x0 = box[i][0] + somma_larghezza
        x1 = box[i][2] + somma_larghezza
        y0=box[i][1]
        y1=box[i][3]

        new_box.append([x0,y0,x1,y1])

    
    return new_box

def get_graph(json_file):
    with open(json_file) as f:
        data = json.load(f)
        bounding_boxes, page,relation,parent, labels, text = get_info_json(data)
        
        n_bb = [(normalize_bounding_box(box, 596, 842)) for box in bounding_boxes ]
        centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in n_bb ]
      #  dim = [((box[2] + box[0]), (box[3] - box[1])) for box in n_bb ]
      #  #distances = cdist(centroids, centroids)  # Matrice distanza con ogni punto

        i, j, labels_edge = get_edge_node(data, bounding_boxes, page, relation, parent)
        #print(set(labels))
        # Graph
        g = dgl.graph((i, j))
        g.edata['label'] = th.tensor(labels_edge)

        #Node Features
        g.ndata['centroids'] = th.tensor(centroids)

       # bb_all = get_all_bb_rotolone(page, bounding_boxes)
       # num_page = page[len(page)-1]

        # n_bb_all = [(normalize_bounding_box(box, 596*num_page, 842)) for box in bb_all ]
        # g.ndata['bb_all'] = th.tensor(n_bb_all)
        g.ndata['bb'] = th.tensor(n_bb)

    #    g.ndata['dim'] = th.tensor(dim)
    #    g.ndata['page'] = th.tensor(page)
        #print(g.nodes())
        # Calcola le coordinate relative per ogni nodo
        relative_coordinates = calculate_relative_coordinates(n_bb)
        g.ndata['relative_coordinates'] = th.tensor(relative_coordinates)

        encoded_labels = processing_lab(labels)
        g.ndata['labels'] = th.tensor(encoded_labels) 

        # node_embeddings = []
        # for i, t in enumerate(text):
        #     embedding = generate_embedding(t)
        #     node_embeddings.append(embedding)
        # g.ndata['embedding'] = th.stack(node_embeddings)
     #   print( g.ndata['embedding'].shape)
        #g = dgl.remove_self_loop(g)
       # g = dgl.add_self_loop(g) # PEr quando usi Gatv2!!
       # g.set_batch_num_nodes(g.batch_num_nodes())
       # g.set_batch_num_edges(g.batch_num_edges())
        #num_edges = g.number_of_edges()
    return g, page, centroids, text

def processing_lab(labels):
    labels = ['other' if label in ['figcap', 'opara', 'secx','tabcap'] else label for label in labels]
    labels = ['meta' if label in ['mail', 'foot', 'title','affili', 'fnote', 'author'] else label for label in labels]

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    return encoded_labels

def get_one_g():
    json = 'data_h/json/ACL_2020.acl-main.99.json'
    g, _, _, _= get_graph(json)
    save('prova',g)
 #   print(g.edata['label'].shape)
    return g

def load(name):
    # load processed data from directory `self.save_path`
    graph_path = name+'_dgl_graph.bin'#os.path.join
    graphs = load_graphs(graph_path)
    return graphs

def save(name,graphs):
    # save graphs and labels
    graph_path = name+ '_dgl_graph.bin'#os.path.join(
    print(graph_path)
    save_graphs(graph_path, graphs)

def get_graphs(type):
    path_json = 'HRDS/' + type +'/'
    list_j = os.listdir(path_json)
    all_graph = []
    pages = []
    centr =[]
    texts = []
    c = 0
    for j in list_j:
        print(c)
        json = path_json + j
        g, page, centroid, text = get_graph(json)
        #print(g)
        all_graph.append(g)
        pages.append(page)
        centr.append(centroid)
        texts.append(text)
        c = c+1
   # print(np.concatenate(pages, axis=0))
    return all_graph, np.concatenate(pages, axis=0), np.concatenate(centr, axis=0), np.concatenate(texts, axis=0)
   
if __name__ == '__main__':
    #get_graphs()
    g = get_one_g()
  #  g = load('prova')
  #  print(g[0])

'''
    {'author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 
    'fig', 'mail', 'secx', 'title', 
    'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}

{meta, contain, connect, equality}

'''

#dataset = GraphDataset(all_graph)
#torch.save(dataset, 'dataset.pt')