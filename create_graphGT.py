# ogni json mi dà 1 grafo ... 

import os
import json 
import dgl
import torch as th
from scipy.spatial.distance import cdist
import dgl.data


def get_info_json(data):
    bounding_boxes = [item['box'] for item in data] # salvo tutte le bb delle miei pagine
    page = [item['page'] for item in data] 
    labels = [item['class'] for item in data]
    relation = [item['relation'] for item in data] 
    parent = [item['parent_id'] for item in data] 
    return bounding_boxes, page,relation,parent, labels

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
                        elif prova == True: #salvo solo 1 rosso
                            array_edges.append([i-k,i])
                            node_i.append(i-k)
                            node_j.append(i)

                            labels_edge.append(0)
                            prova = False
                k = k + 1
            # array_edges
        return node_i, node_j, labels_edge

def get_graph(json_file):
    with open(json_file) as f:
        data = json.load(f)
        bounding_boxes, page,relation,parent, labels  = get_info_json(data)
        
        centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in bounding_boxes ]
       
        #distances = cdist(centroids, centroids)  # Matrice distanza con ogni punto

        i, j, labels_edge = get_edge_node(data, bounding_boxes, page, relation, parent)
        
        # Graph
        g = dgl.graph((i, j))
        g.edata['label'] = th.tensor(labels_edge)

        #Node Features
        g.ndata['centroids'] = th.tensor(centroids)
        g.ndata['bb'] = th.tensor(bounding_boxes)
        g.ndata['page'] = th.tensor(page)
        # One hot encoder 
        #g.ndata['labels'] = th.tensor(labels) 
        #print(dgl.has_self_loop(g))
        #g = dgl.remove_self_loop(g)
       # g = dgl.add_self_loop(g) # PEr quando usi Gatv2!!
       # g.set_batch_num_nodes(g.batch_num_nodes())
       # g.set_batch_num_edges(g.batch_num_edges())
        #num_edges = g.number_of_edges()
    return g

def get_one_g():
    json = 'data_h/json/ACL_2020.acl-main.99.json'
    g = get_graph(json)
    return g


def get_graphs(type):
    path_json = 'HRDS/' + type +'/'
    list_j = os.listdir(path_json)
    all_graph = []
    for j in list_j:
        json = path_json + j
        g = get_graph(json)
        #print(g)
        all_graph.append(g)
    return all_graph
   
if __name__ == '__main__':
    get_graphs()

'''
    {'author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 
    'fig', 'mail', 'secx', 'title', 
    'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}

{meta, contain, connect, equality}

'''

#dataset = GraphDataset(all_graph)
#torch.save(dataset, 'dataset.pt')