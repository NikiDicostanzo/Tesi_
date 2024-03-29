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
import numpy as np


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
def get_edge_node(data, bounding_boxes, page, relation, parent, num_arch_node, class3):
        labels_edge = []
        array_edges = []
        node_i = []
        node_j = []
        #num_arch_node = 1
        for i in range(len(data)): 
            k = 1
            break_count = 0 # voglio avere almeno 3 grafi per ogni arco
            plot_flow = True
            while k< 16 and i - k >=0: #and k<i+2 :
                if page[i] == page[i-k]:
                    # Stesso blocco 
                    #print(title_condition(data, i, i-k), data[i], data[i-k])
                    if i >0 and relation[i]=='connect' and parent[i] == data[i-k]['line_id']:#TODO and (data[i]['class'] != 'equ' and data[i-k]['class'] != 'equ'):
                        #array_edges.append([i-k,i]) # arco con quello precedente
                        add_edge(labels_edge, node_i, node_j, i, k, 1) #BLUE
                        plot_flow = False
                        if break_count > num_arch_node:
                            break
                        break_count = break_count + 1
                    elif class3 and plot_flow == True and data[i]['class'] != data[i-k]['class'] and (title_condition(data, i, i-k) or title_condition(data, i-k, i)): # Quello successivo
                        add_edge(labels_edge, node_i, node_j, i, k, 2) #CYNEùìù
                        plot_flow = False # di precedente ne ha solo uno, una volta che lo trova stop
                        break
                    elif break_count < num_arch_node:
                        #print(title_condition(data, i, i-k), data[i]['class'], data[i-k]['class'])
                        add_edge(labels_edge, node_i, node_j, i, k, 0)#RED
                    
                        if data[i]['is_meta'] ==True or data[i]['class'] in ['fig', 'other']:
                            break
                        break_count = break_count + 1
                    
                else: # Collegamento tra pagine diverse
                    
                    if i >0  and data[i-k]['is_meta'] !=True:# and k>2:
                        if relation[i]=='connect' and parent[i] == data[i-k]['line_id']:
                          #  array_edges.append([i-k,i]) # arco con quello precedente
                            add_edge(labels_edge, node_i, node_j, i, k, 1)
                            plot_flow = False
                            if break_count > num_arch_node:
                                break
                            break_count = break_count + 1
                           #Titolo  # Quello precedente
                        elif class3 and plot_flow == True and (title_condition(data, i, i-k)or title_condition(data, i-k, i)): # Quello successivo
                            add_edge(labels_edge, node_i, node_j, i, k, 2)
                            plot_flow = False
                            break
                        elif break_count < num_arch_node:#prova == True: #salvo solo 1 rosso
                            add_edge(labels_edge, node_i, node_j, i, k, 0)
                            if data[i]['is_meta'] == True or data[i]['class'] in ['fig', 'tab', 'tabcap', 'opara', 'figcap']:#in yolo ho stolo other :|
                                break
                            break_count = break_count + 1
                k = k + 1
            # array_edges
            
        return node_i, node_j, labels_edge

def add_edge(labels_edge, node_i, node_j, i, k, type_edge):
    node_i.append(i-k)
    node_j.append(i)
    labels_edge.append(type_edge) 

def title_condition(labels, s, t):
    return (labels[s]['class'] in ['sec1','sec2','sec3', 'para', 'equ'] and labels[t]['class'] in ['para', 'equ','fstline', 'sec1','sec2','sec3'])

# def title_condition(labels_yolo, s, t):
#     return (labels_yolo[s] in ['sec1','sec2','sec3', 'para', 'equ'] and labels_yolo[t] in ['para', 'equ','fstline', 'sec1','sec2','sec3'])

# def calculate_relative_coordinates(bb, k):#=5):
#     # Assumendo che g.ndata['bb'] contenga le bounding boxes normalizzate
#     relative_coordinates = []
#     for i in range(len(bb)):
       
#         current_bb = np.array(bb[i])
#         if i + k < len(bb):
#             next_bbs = np.array(bb[i+1:i+k+1])
#         else:
#             next_bbs = np.zeros(k) #np.array([]) #TODO rivedere
#         # Calcola la differenza relativa tra la bounding box corrente e quelle dei k vicini successivi
#        # relative_diffs = [np.abs(current_bb - next_bb) for next_bb in next_bbs]
#         relative_diffs = [abs(next_bb - current_bb) for next_bb in next_bbs]

#         relative_coordinates.append(relative_diffs)
#     return np.array(relative_coordinates)

def calculate_relative_coordinates(bb, k):
    relative_coordinates = []
    for i in range(len(bb)):
        current_bb = np.array(bb[i])
        # Calcola i k vicini precedenti
        if i - k >= 0:
            prev_bbs = np.array(bb[i-k:i])
        else:
            prev_bbs = np.zeros(k) # o qualsiasi valore di default
        # Calcola i k vicini successivi
        if i + k < len(bb):
            next_bbs = np.array(bb[i+1:i+k+1])
        else:
            next_bbs = np.zeros(k) # o qualsiasi valore di default
        
        # Calcola la differenza relativa tra la bounding box corrente e quelle dei k vicini precedenti e successivi
        relative_diffs_prev = [abs(prev_bb - current_bb) for prev_bb in prev_bbs]
        relative_diffs_next = [abs(next_bb - current_bb) for next_bb in next_bbs]
        
        # Combina le differenze relative dei vicini precedenti e successivi
        relative_diffs = np.concatenate([relative_diffs_prev, relative_diffs_next], axis = 0)
        
        relative_coordinates.append(relative_diffs)

       # print(relative_diffs)#, relative_coordinates)
    return relative_coordinates

def normalize_bounding_box(box, image_width, image_height):
    # Normalizzazione diretta delle coordinate x e y
    normalized_x0 = box[0] / image_width
    normalized_y0 = box[1] / image_height
    normalized_x1 = box[2] / image_width
    normalized_y1 = box[3] / image_height
    
    return normalized_x0, normalized_y0, normalized_x1, normalized_y1


def processing_lab(labels):
    #print(set(labels))
    labels = ['other' if label in ['figcap', 'opara', 'secx','tabcap', 'alg'] else label for label in labels]
    labels = ['meta' if label in ['mail', 'foot', 'title','affili', 'fnote', 'author'] else label for label in labels]

    #le = LabelEncoder()
    #encoded_labels = le.fit_transform(labels)
    ohe = OneHotEncoder(sparse=False)
  #  print(set(labels))
    # Adatta e trasforma le etichette
    encoded_labels = ohe.fit_transform(np.array(labels).reshape(-1, 1))
    #print(encoded_labels)
    return encoded_labels

def calculate_relative_y(bb, k):
    relative_y = []
    for i in range(len(bb)):
        current_bb = np.array(bb[i])
        # Estrai solo la coordinata y (secondo valore) della bounding box corrente
        current_y = current_bb[1]
        
        # Calcola i k vicini precedenti
        if i - k >= 0:
            prev_bbs = np.array(bb[i-k:i])
        else:
           # prev_bbs = np.zeros([k]) # o qualsiasi valore di default
            prev_bbs = []
            for pb in range(k): #TODO
                prev_bbs.append(np.zeros([4]))
       # print(prev_bbs, '|',current_y )
        # Calcola i k vicini successivi
        if i + k < len(bb):
            next_bbs = np.array(bb[i+1:i+k+1])
        else:
            #next_bbs = np.zeros(k) # o qualsiasi valore di default
            next_bbs = []
            for nb in range(k): #TODO
                next_bbs.append(np.zeros([4]))
        
        # Calcola la differenza relativa tra la coordinata y della bounding box corrente e quelle dei k vicini precedenti e successivi
        relative_diffs_prev = [abs(prev_bb[1] - current_y) for prev_bb in prev_bbs]
        relative_diffs_next = [abs(next_bb[1] - current_y) for next_bb in next_bbs]
        
        # Combina le differenze relative dei vicini precedenti e successivi
        relative_diffs = np.concatenate([relative_diffs_prev, relative_diffs_next], axis = 0)
        
        relative_y.append(relative_diffs)
        #print(len(relative_diffs), (relative_diffs))
    return relative_y

def calculate_relative_labels(labels_hot, k):
    relative_labels = []
    for i in range(len(labels_hot)):
        # Assuming labels are numerical for simplicity
        current_label = np.array(labels_hot[i])
        
        # Calculate k nearest neighbors
        if i - k >= 0:
            prev_labels = np.array(labels_hot[i-k:i])
        else:
            prev_labels = []
            for _ in range(k):
                prev_labels.append(np.full_like(current_label, -1))#np.zeros_like(current_label))
        
        if i + k < len(labels_hot):
            next_labels = np.array(labels_hot[i+1:i+k+1])
        else:
            next_labels = []
            for _ in range(k):
                next_labels.append(np.zeros_like(np.full_like(current_label, -1)))#current_label))
        
        # Combine the labels of the k nearest neighbors
        relative_labels.append(np.concatenate([prev_labels, next_labels], axis=0))
    
    return relative_labels

def weighted_labels(bb, labels, kr):
    # Calcola le coordinate y
    distances = np.array(calculate_relative_y(bb, kr)) # k vicini -< 10
    
    # Processa le labels
    encoded_labels = processing_lab(labels) # classi 9

    # 0 vuol dire che non ci sono k elementi prima
    relative_labels = np.array(calculate_relative_labels(encoded_labels, kr))
 
    relative_distance_reshaped = distances.reshape(-1, kr*2, 1)

    # Moltiplica relative_labels e relative_distance_reshaped
    weighted_labels = relative_labels * relative_distance_reshaped

    # Somma lungo l'ultimo asse per aggregare i risultati pesati
    aggregated_labels = np.sum(weighted_labels, axis=2)
    return aggregated_labels

def load(name):
    # load processed data from directory `self.save_path`
    graph_path = name+'_dgl_graph.bin'#os.path.join
    graphs = load_graphs(graph_path)
    return graphs

def save(name,graphs):
    # save graphs and labels
    graph_path = name+ '_dgl_graph.bin'#os.path.join(
   # print(graph_path)
    save_graphs(graph_path, graphs)

def get_graphs_gt(type, kr, num_arch_node, class3):
    path_json = 'HRDS/' + type +'/'
    list_j = os.listdir(path_json)
    all_graph = []
    pages = []
    centr =[]
    texts = []
    c = 0
    for j in list_j:
        #print(c)
        json = path_json + j
        g, page, centroid, text = get_graph_3class(json, kr, num_arch_node, class3)
        all_graph.append(g)
        pages.append(page)
        centr.append(centroid)
        texts.append(text)
        c = c+1
    return all_graph, np.concatenate(pages, axis=0), np.concatenate(centr, axis=0), np.concatenate(texts, axis=0)
   

def get_graph_3class(json_file, kr, num_arch_node, class3):
    with open(json_file) as f:
        data = json.load(f)
        bounding_boxes, page,relation,parent, labels, text = get_info_json(data)

        n_bb = [(normalize_bounding_box(box, 596, 842)) for box in bounding_boxes ]
        centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in n_bb ]
      #  dim = [((box[2] + box[0]), (box[3] - box[1])) for box in n_bb ]
      #  #distances = cdist(centroids, centroids)  # Matrice distanza con ogni punto

        i, j, labels_edge = get_edge_node(data, bounding_boxes, page, relation, parent, num_arch_node, class3)
      #  print(set(labels_edge))
        # Graph
        g = dgl.graph((i, j))
        g.edata['label'] = th.tensor(labels_edge)

        #Node Features
        g.ndata['centroids'] = th.tensor(centroids)

        # node_embeddings = []
        # for i, t in enumerate(text):
        #     embedding = generate_embedding(t)
        #     node_embeddings.append(embedding)
        # g.ndata['embedding'] = th.stack(node_embeddings)

       # num_page = page[len(page)-1]

        g.ndata['bb'] = th.tensor(n_bb)

    #    g.ndata['dim'] = th.tensor(dim)
      #  g.ndata['page'] = th.tensor(page)
        # Calcola le coordinate relative per ogni nodo
        relative_coordinates = calculate_relative_coordinates(n_bb, kr)
        g.ndata['relative_coordinates'] = th.tensor(relative_coordinates)

        encoded_labels = processing_lab(labels)
        g.ndata['labels'] = th.tensor(encoded_labels) 

    return g, page, centroids, text

def get_graph_merge(i, j, labels_edge, bounding_boxes, labels, page ):
   
   # i, j, labels_edge, bounding_boxes, labels, page = get_graph()
   # 
    n_bb = [(normalize_bounding_box(box, 596, 842)) for box in bounding_boxes ]
    centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in n_bb ]
    
    # Graph
    g = dgl.graph((i, j))
    g.edata['label'] = th.tensor(labels_edge)

    #Node Features
    g.ndata['centroids'] = th.tensor(centroids)

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
  
    return g, page, centroids

   
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