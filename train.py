import torch
from create_graphGT import get_one_g, get_graphs

from create_graphGT_3lab import get_graphs3
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

def model_train(graph, val_graph, name_model, epoch):
    print(graph)

    node_features, input, edge_label = get_nfeatures(graph)
    print(type(edge_label))
    #node_features_val, input_val, edge_label_val = get_nfeatures(val_graph)
    a = np.array(edge_label)
    unique_values, counts = np.unique(a, return_counts=True)
    print(unique_values, counts)  #[0 1 2] [ 53094 587187 106568]
    print('node_features:', node_features.shape, '|', 'input', input, '|','edge_label', edge_label.shape)
   
    out_features = 3
    hidden = 20
    #model = EdgeClassifier(graph.num_edges(), 1, 0.2, node_features, 300, hidden, device, False)
    model = Model(input, hidden , out_features).to(device)
    opt = torch.optim.Adam(model.parameters())

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
    # Salva il modello addestrato
    torch.save(model.state_dict(), name_model)

def get_nfeatures(graph):

  #  page = graph.ndata['page'].float().unsqueeze(-1)
    centroids = graph.ndata['centroids'] 
    bb = graph.ndata['bb']   
   # dim = graph.ndata['dim']   
  #  rotbb = graph.ndata['bb_all']   
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
    
def main_train(model_name, epoch):
    #bg_train = get_one_g()#get_graphs()
  #  train_graphs, _, _ , _= get_graph_merge_gt()
    train_graphs, _, _ , _= get_graphs3('train')
    print('Start Train')

    #train_graphs = get_graph()
  #  train_graphs, val_graphs = train_test_split(graph_train, test_size=0.01)
    
    bg_train = dgl.batch(train_graphs) # num_nodes=725391, num_edges=811734,
    bg_train = bg_train.int().to(device)

    bg_val = ''
  #  bg_val = dgl.batch(val_graphs) # num_nodes=7523, num_edges=8389,
  #  bg_val = bg_val.int().to(device)
  #  print(bg_train, '\n', bg_val)

    model_train(bg_train, bg_val, model_name, epoch)

if __name__ == '__main__':
    main_train('model_bb_lab_cent_rel6_3class.pth', 500)
   
   # model_test('model_no_page.pth')
    
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