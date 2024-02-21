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
    node_features = graph.ndata['page'].float()
    edge_label = graph.edata['label']
    print(node_features)
    # in_feature = data.node_num_classes
    out_features = 2 
    hidden = 20
    #model = Model(graph.num_nodes, hidden , out_features)
    inputs = node_features.unsqueeze(-1)
    print(inputs)
    model = Model(1, 32,5)
    edge_label = edge_label.unsqueeze(-1)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        pred = model(graph, inputs)         
        loss = ((pred - edge_label) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())

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