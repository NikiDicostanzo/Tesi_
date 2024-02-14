from GatModel import GATModel 
import torch
from graph import get_graph
import os


def my_graph():
    json_file = 'data_h/json/ACL_2020.acl-main.99.json'
    path_save = 'data_h/save/'
    folder = 'data_h/image/'
    list_image = os.listdir(folder)
   # print(list_image)
    #for page in range(len(list_image)):
    page = 0
    path_image = folder + list_image[page]
    image_save = path_save + list_image[page]
    #get_near(json_file, page, path_image, image_save)
    g = get_graph(json_file, page, path_image, image_save)
    return g

def main():
    #model = GATModel(input_dim, output_dim, num_heads)
    graph = my_graph()

    node_features = graph.ndata['centroids']#.float()
    edge_label = graph.edata['label']#.long()

    #node_features = edge_pred_graph.ndata['feature']
    #edge_label = edge_pred_graph.edata['label']
    #train_mask = edge_pred_graph.edata['train_mask']

    model = GATModel(2, 23, 4)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        pred = model(graph, node_features)
        #loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()
        loss = ((pred - edge_label) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
    
if __name__ == '__main__':
    main()