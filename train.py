import torch
from plot_edge_multipage_YOLO_new import get_graph_yolo
from create_graphGT import get_one_g, get_graphs

from create_graphGT_3lab import get_graphs_gt
import numpy as np

from plot_edge_multipage_GT_Merge import get_graph_merge_gt
import os
import torch.nn.functional as F
import dgl
from sklearn.model_selection import train_test_split
#from plot_edge_multipage_YOLO import get_graph
#from Tesi_.model2_git import EdgeClassifier
from model import Model
from model3 import Model3

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

def model_train(graph, val_graph, name_model, epoch, num_outc):
    print(graph)

    node_features, input, edge_label = get_nfeatures(graph)
    print(type(edge_label))
    #node_features_val, _, edge_label_val = get_nfeatures(val_graph)

    #node_features_val, input_val, edge_label_val = get_nfeatures(val_graph)
    a = np.array(edge_label)
    unique_values, counts = np.unique(a, return_counts=True)
    print(unique_values, counts)  #[0 1 2] [ 53094 587187 106568]
    print('node_features:', node_features.shape, '|', 'input', input, '|','edge_label', edge_label.shape)
   
    out_features = num_outc
    hidden = 20
    #model = EdgeClassifier(graph.num_edges(), 1, 0.2, node_features, 300, hidden, device, False)
    model = Model(input, hidden , out_features).to(device)
  #  model = Model3(input, hidden , out_features, 1).to(device)

    opt = torch.optim.Adam(model.parameters())#, lr=1e-08)
    #opt = torch.optim.SGD(model.parameters(), lr=0.1)

    min_val_loss = 100000000
    for epoch in range(epoch):
        logit = model(graph, node_features)   
        #print(logit, '/n',edge_label.squeeze())
        loss = F.cross_entropy(logit, edge_label.squeeze())

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        acc = accuracy(logit, edge_label)
        if epoch % 10 == 0:
             print('Epoch {:05d} | Loss {:.4f} | Accuracy train set {:.4f}'
                   .format(epoch, loss.item(), acc))
             
        # Calcola la perdita e l'accuratezza sul set di validazione
        # model.eval() # Imposta il modello in modalità di valutazione
        # with torch.no_grad(): # Disabilita il calcolo del gradiente per la valutazione
        #     val_logit = model(val_graph, node_features_val)
        #     val_loss = F.cross_entropy(val_logit, edge_label_val.squeeze())
        #     val_acc = accuracy(val_logit, edge_label_val)
        #     if epoch % 10 == 0:
        #       print('Validation Loss: {:.4f} | Validation Accuracy: {:.4f}'
        #             .format(val_loss.item(), val_acc))

        # # Salva il modello se mostra le migliori prestazioni sul set di validazione
        # if val_loss < min_val_loss:
        #     min_val_loss = val_loss
        #     torch.save(model.state_dict(), 'best_model.pth')
    # Salva il modello addestrato
    torch.save(model.state_dict(), name_model)

def get_nfeatures(graph):

  #  page = graph.ndata['page'].float().unsqueeze(-1)
    centroids = graph.ndata['centroids'] 
    bb = graph.ndata['bb']      
    lab = graph.ndata['labels'].float().unsqueeze(1)
    coord_rel = graph.ndata['relative_coordinates'].float() 
    coord_rel_reshaped = coord_rel.view(coord_rel.size(0), -1)
  
  #  text_emb = graph.ndata['embedding']
    
  #  Concatena 'pages',centroid, 'bbs' lungo la dimensione delle caratteristiche
    #bb, lab, coord_rel_reshaped, page, centroids, text_emb
    node_features = torch.cat([bb, lab, centroids, coord_rel_reshaped], dim=-1) #

    node_features = node_features.to(device)
    print('node_feature:', node_features.shape)
    input = node_features.shape[1]
    edge_label = graph.edata['label'].unsqueeze(-1)
    return node_features,input,edge_label

def accuracy(indices, labels):
    if labels.dim() >  1:
        labels = labels.squeeze()
    indices = indices.argmax(dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() *1.0/ len(labels)

def model_test(model_name):
    graph_test, _ , _ ,_ = get_graphs3('test')
    #graph_test = get_graph()
    
    graph_test = dgl.batch(graph_test) # num_nodes=725391, num_edges=811734,
    graph_test = graph_test.int().to(device)

    node_features, input, edge_label = get_nfeatures(graph_test)

    out_features = 3 
    hidden = 20

    # Carica il modello addestrato
    model = Model(input, hidden , out_features).to(device)
    model.load_state_dict(torch.load(model_name))

    # Imposta il modello in modalità di valutazione
    model.eval()

        # Fai le previsioni sul set di test
    with torch.no_grad():
        logits = model(graph_test, node_features)
        _, predictions = torch.max(logits, dim=1)

    # Confronta le previsioni con le etichette di classe effettive
    
    acc = accuracy(logits, edge_label)

    print('Test Accuracy: {:.4f}'.format(acc)) # Test Accuracy: 0.9009
    
def main_train(model_name, epoch, kr, num_arch_node, exp, class3):
    #bg_train = get_one_g()#get_graphs()
  #  train_graphs, _, _ , _= get_graph_merge_gt()
    if exp == 'yolo':
      train_graphs, _ = get_graph_yolo(kr, num_arch_node, class3, 'train')
      #val_graphs, _ = get_graph_yolo(kr, num_arch_node, class3, 'val')
    else:
      train_graphs, _, _ , _= get_graphs_gt('train', kr, num_arch_node, class3)
    #  train_graphs, _, _ , _= get_graphs_gt('train', kr, num_arch_node, class3)
    
    print('Start Train')
    bg_train = dgl.batch(train_graphs) # num_nodes=725391, num_edges=811734,
    bg_train = bg_train.int().to(device)

    bg_val = ''#dgl.batch(val_graphs) # num_nodes=7523, num_edges=8389,
    #bg_val = bg_val.int().to(device)
    print(bg_train, '\n', bg_val)

    model_train(bg_train, bg_val, model_name, epoch, 3) #TODO num_outc-> 3class == False

if __name__ == '__main__':
    main_train('model_bb_lab_cent_5rel5_3class_k2_yolo_mmm.pth', 700, 5, 2, 'yolo', True) 
    #kr = num_dist_rel || num_arch_node = # edge x nodo | 3class = true

   # model_test('model_no_page.pth')
    

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