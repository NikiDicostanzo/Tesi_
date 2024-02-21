import torch
from create_graphGT import get_one_g, get_graphs
import os
import torch.nn.functional as F

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
def model_train(graph):
    print(graph)

    #torch.Size([902, 4]) torch.Size([902, 2]) torch.Size([902])
    node_features, input, edge_label = get_nfeatures(graph)
    # in_feature = data.node_num_classes
    out_features = 2 
    hidden = 20
    model = Model(input, hidden , out_features)
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        _, pred = model(graph, node_features)         
        loss = ((pred - edge_label) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        #print(loss.item())
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f}')

def get_nfeatures(graph):
    #labels_node = 
    page = graph.ndata['page'].float().unsqueeze(-1)
    centroids = graph.ndata['centroids'] 
    bb = graph.ndata['bb']    

    # Concatena 'pages',centroid, 'bbs' lungo la dimensione delle caratteristiche
    node_features = torch.cat([page, centroids], dim=-1)
    node_features = torch.cat([node_features, bb], dim=-1)
    #print(node_features)

    input = node_features.shape[1]
    edge_label = graph.edata['label'].unsqueeze(-1)
    #print(bb.shape, centroids.shape, page.shape)
    return node_features,input,edge_label

def main():
    graph = get_one_g()#get_graphs()
    model_train(graph)

if __name__ == '__main__':
    main()

   # print(get_one_g())
    
""" 

tuo tensore di input dovrebbe avere una dimensione di (num_nodes, in_feats) 
dove num_nodes è il numero di nodi nel tuo grafo e in_feats è il numero totale 
di caratteristiche per nodo.
Poiché 'bb' ha 4 valori e 'page' ha 1 valore, il numero totale di caratteristiche per nodo sarà 5
Per preparare l'input corretto, devi combinare i tensori delle caratteristiche 'bb' e 'page' 
in un unico tensore
"""