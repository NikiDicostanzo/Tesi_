import torch
from create_graphGT import get_one_g, get_graphs
import os
import torch.nn.functional as F
import dgl

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

def model_train(graph):
    print(graph)

    #torch.Size([902, 4]) torch.Size([902, 2]) torch.Size([902])
    node_features, input, edge_label = get_nfeatures(graph)
    # in_feature = data.node_num_classes
    out_features = 2 
    hidden = 20
    model = Model(input, hidden , out_features).to(device)
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(1000):
        _, pred = model(graph, node_features)         
        loss = ((pred - edge_label) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        #acc = evaluate(model, graph, node_features, edge_label)
        #acc2 = accuracy(pred, edge_label)
        if epoch % 10 == 0:
            print(epoch, loss.item())
            # print('Epoch {:05d} | Loss {:.4f} | Accuracy w/ Validation data set {:.4f}|{:.4f} '
            #       .format(epoch, loss.item(), acc, acc2))

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

# evaluate model by accuracy
def evaluate(model, graph, features, labels):
    model.eval()
    with torch.no_grad():
        logits, _ = model(graph, features)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def main():
    #graph = get_one_g()#get_graphs()

    graph_train = get_graphs()
    bg = dgl.batch(graph_train)
    bg = bg.int().to(device)
    
    model_train(bg)

if __name__ == '__main__':
    main()
    
   # print(get_one_g())
    
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