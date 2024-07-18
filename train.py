import torch
#from plot_edge_multipage_YOLO_new import get_graph_yolo
from plot_edge_multipage_PARSE import get_graph_parse
from z_boh.plot_edge_multipage_YOLO5 import get_graph_yolo

from create_graphGT import get_one_g, get_graphs

from create_graphGT_3lab import get_graphs_gt
import numpy as np

from z_boh.plot_edge_multipage_GT_Merge import get_graph_merge_gt
import os
import torch.nn.functional as F
import dgl
from sklearn.model_selection import train_test_split
#from plot_edge_multipage_YOLO import get_graph
#from Tesi_.model2_git import EdgeClassifier
from model import Model

# Eseguire train del modello (devo crearlo )
# Batch dei grafi !!
# ... 
# Ora faccio su GT 
# Poi devo fare su YOLO (prima devo fare mapping !!! e embedding testo)

# Graph(num_nodes=902, num_edges=1895,
#       ndata_schemes={'page': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={'label': Scheme(shape=(), dtype=torch.int64)})
# Provo il modello su GT 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def model_train(graph, val_graph, name_model, epoch, num_outc, array_features):
    print(graph)

    node_features, input, edge_label = get_nfeatures(graph, array_features)
    print(type(edge_label))
    #node_features_val, _, edge_label_val = get_nfeatures(val_graph)

    #node_features_val, input_val, edge_label_val = get_nfeatures(val_graph)
    a = np.array(edge_label)
    unique_values, counts = np.unique(a, return_counts=True)
    print(unique_values, counts)  #[0 1 2] [ 53094 587187 106568]
    print('node_features:', node_features.shape, '|', 'input', input, '|','edge_label', edge_label.shape)
   
    out_features = num_outc
    hidden = 20

    model = Model(input, hidden , out_features).to(device)

    opt = torch.optim.Adam(model.parameters())#, lr=1e-08)
    best_acc = 0
    for epoch in range(epoch):
        logit = model(graph, node_features)   
        #print(logit, '/n',edge_label.squeeze())
        loss = F.cross_entropy(logit, edge_label.squeeze())

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        acc = accuracy(logit, edge_label)
        if epoch % 30 == 0:
             print('Epoch {:05d} | Loss {:.4f} | Accuracy train set {:.4f}'
                   .format(epoch, loss.item(), acc))
        if best_acc < acc:
           best_acc = acc
          # print('EPOCH', epoch)
           my_model = model
           if epoch > 500:
            print('BEST Epoch {:05d} | Loss {:.4f} | Accuracy train set {:.4f}'
                    .format(epoch, loss.item(), best_acc))
           #torch.save(model.state_dict(), name_model)
  
    # Salva il modello addestrato
    print('MODEL: ', name_model)
    torch.save(my_model.state_dict(), name_model)



def accuracy(indices, labels):
    if labels.dim() >  1:
        labels = labels.squeeze()
    indices = indices.argmax(dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() *1.0/ len(labels)

def model_test(model_name, epoch, kr, num_arch_node, num_arch_node2, exp, class3, array_features):
    
    graph_test, page = get_graph_parse(kr, num_arch_node, num_arch_node2, class3, 'test', array_features)    
    #graph_test, _, _ , _ = get_graphs_gt('test', kr, num_arch_node, class3, array_features)#get_graphs3('test')
    #graph_test = get_graph()

    dimensione_desiderata = 11 # TODO numero delle classi

    for graph in graph_test:
      #print(graph.ndata['labels'].shape[1])
      # Assicurati che tutte le feature dei nodi abbiano la stessa dimensione e tipo di dati
      if 'labels' not in graph.ndata or graph.ndata['labels'].shape[1] != dimensione_desiderata:
          # Crea una feature di placeholder con la dimensione desiderata
          placeholder = torch.zeros((graph.number_of_nodes(), dimensione_desiderata), dtype=torch.float64)
          graph.ndata['labels'] = placeholder
    
    graph_test = dgl.batch(graph_test) # num_nodes=725391, num_edges=811734,
    graph_test = graph_test.int().to(device)
    

    node_features, input, edge_label = get_nfeatures(graph, array_features)

    out_features = 3 #num_outc
    hidden = 20

    model = Model(input, hidden , out_features).to(device)

    # Carica il modello addestrato
    model.load_state_dict(torch.load(model_name))

    # Imposta il modello in modalità di valutazione
    model.eval()

        # Fai le previsioni sul set di test
    with torch.no_grad():
        
        logits = model(graph_test, node_features)
        _, predictions = torch.max(logits, dim=1)

    # Confronta le previsioni con le etichette di classe effettive
    
    acc = accuracy(logits, edge_label)

    print('Test Accuracy: {:.4f}'.format(acc)) # Test Accuracy
    
def main_train(model_name, epoch, kr, num_arch_node, num_arch_node2, exp, class3, array_features):
    #bg_train = get_one_g()#get_graphs()
  #  train_graphs, _, _ , _= get_graph_merge_gt()
    if exp == 'yolo':
      train_graphs, _ = get_graph_yolo(kr, num_arch_node, class3, 'train', array_features)
      #val_graphs, _ = get_graph_yolo(kr, num_arch_node, class3, 'val')
    elif exp == 'parse':
        train_graphs, page = get_graph_parse(kr, num_arch_node, num_arch_node2, class3, 'train', array_features)    
    else:
      train_graphs, _, _ , _= get_graphs_gt('train', kr, num_arch_node, class3, array_features)
    #  train_graphs, _, _ , _= get_graphs_gt('train', kr, num_arch_node, class3)
    
    dimensione_desiderata = 11 # TODO numero delle classi
    dimensione_desiderata_font = 6

    for graph in train_graphs:
      # Assicurati che tutte le feature dei nodi abbiano la stessa dimensione e tipo di dati
      if 'labels' not in graph.ndata or graph.ndata['labels'].shape[1] != dimensione_desiderata:
          # Crea una feature di placeholder con la dimensione desiderata
          placeholder = torch.zeros((graph.number_of_nodes(), dimensione_desiderata), dtype=torch.float64)
          graph.ndata['labels'] = placeholder
      if 'font' not in graph.ndata or graph.ndata['font'].shape[1] != dimensione_desiderata_font:
          # Crea una feature di placeholder con la dimensione desiderata
          placeholder = torch.zeros((graph.number_of_nodes(), dimensione_desiderata_font), dtype=torch.float64)
          graph.ndata['font'] = placeholder

    print('Start Train')
    bg_train = dgl.batch(train_graphs) # num_nodes=725391, num_edges=811734,
    bg_train = bg_train.int().to(device)

    bg_val = ''#dgl.batch(val_graphs) # num_nodes=7523, num_edges=8389,
    #bg_val = bg_val.int().to(device)
    print(bg_train, '\n', bg_val)

    model_train(bg_train, bg_val, model_name, epoch, 3, array_features) #TODO num_outc-> 3class == False

# model_bb_lab_cent_kr66_3class_k2_agg_yolo5

def get_nfeatures(graph, array_features):

  #  page = graph.ndata['page'].float().unsqueeze(-1)
    data_features = []
   # centroids = graph.ndata['centroids']
   # data_features.append(centroids)

   # block = graph.ndata['block'].float().unsqueeze(-1) 
   #data_features.append(block)

    bb = graph.ndata['bb']      
    data_features.append(bb)

    if 'font' in  array_features:
        font = graph.ndata['font'].float()
        data_features.append(font)
 

    if 'lab' in array_features:
      lab = graph.ndata['labels'].float()#.unsqueeze(1)
      data_features.append(lab)

    if 'rel' in array_features:
      coord_rel = graph.ndata['relative_coordinates'].float() 
      coord_rel_reshaped = coord_rel.view(coord_rel.size(0), -1)
      data_features.append(coord_rel_reshaped)

      if 'pageRel' in array_features:
        page_rel = graph.ndata['relative_page'].float()
        page_rel_reshaped = page_rel.view(page_rel.size(0), -1)
        data_features.append(page_rel_reshaped)

    if 'agg' in array_features:
      aggregated_labels =  graph.ndata['aggregated_labels'].float() 
      data_features.append(aggregated_labels)
      
    if 'area' in array_features:
      areas_array = graph.ndata['area'].float().unsqueeze(-1) 
      data_features.append(areas_array)

    if 'w' in array_features:  
      widths = graph.ndata['widths'].float().unsqueeze(-1)
      data_features.append(widths)

    if 'h' in array_features:
      heights = graph.ndata['heights'].float().unsqueeze(-1) 
      data_features.append(heights)

  #  text_emb = graph.ndata['embedding']
    
  #  Concatena 'pages',centroid, 'bbs' lungo la dimensione delle caratteristiche
    #bb, lab, coord_rel_reshaped, page, centroids, text_emb
    node_features = torch.cat(data_features, dim=-1) ##, , aggregated_labels]

    #node_features = torch.cat([widths, heights, bb, centroids, areas_array], dim=-1) ##, , aggregated_labels]
    #node_features = torch.cat([bb, lab, centroids], dim=-1) ##, , aggregated_labels]

    node_features = node_features.to(device)
    print('node_feature:', node_features.shape)
    input = node_features.shape[1]
    edge_label = graph.edata['label'].unsqueeze(-1)
    return node_features,input,edge_label

if __name__ == '__main__':
    #'bb', 'cent', 'lab', 'area', 'w', 'h', 'rel', 'agg'
    array_features = ['bb', 'area', 'w', 'h','cent', 'lab', 'rel', 'agg', 'pageRel']#
    main_train('z_check_FIN/model_bb_area_w_h_cent_lab_rel3_agg_pageRel_in1_out1.pth', 1500, 3, 1, 2, 'parse', True, array_features) 
  #  model_test('z_check_FIN/model_bb_area_w_h_cent.pth', 1500, 6, 1, 1, 'parse', True, array_features) 

  #  #kr = num_dist_rel || num_arch_node = # edge x nodo | 3class = true


#BEST Epoch 01499 | Loss 0.5193 | Accuracy train set 0.8719
   # model_test('model_bb_cent_area_w_h_rel6_lab_agg.pth', 6, 2, True, array_features)
    #kr, num_arch_node, class3,

# k2 -> [699346 586447 106252]
""" 
Graph(num_nodes=732914, num_edges=820123,
      ndata_schemes={'centroids': Scheme(shape=(2,), dtype=torch.float32), 'bb': Scheme(shape=(4,), dtype=torch.int64), 'page': Scheme(shape=(), dtype=torch.int64)}
      edata_schemes={'label': Scheme(shape=(), dtype=torch.int64)})

tuo tensore di input dovrebbe avere una dimensione di (num_nodes, in_feats) 
dove num_nodes è il numero di nodi nel tuo grafo e in_feats è il numero totale 
di caratteristiche per nodo.
Poiché 'bb' ha 4 valori e 'page' ha 1 valore, il numero totale di caratteristiche per nodo sarà 5
Per preparare l'input corretto, devi combinare i tensori delle caratteristiche 'bb' e 'page' 
in un unico tensore
"""