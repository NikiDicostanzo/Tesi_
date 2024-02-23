import torch
from model import Model
from create_graphGT import get_graphs
import os
import dgl
from PIL import Image, ImageDraw as D
from train import get_nfeatures
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def inference(graph_test):
    node_features, input, edge_label = get_nfeatures(graph_test)

    out_features = 2 
    hidden = 20

    # Carica il modello addestrato
    model = Model(input, hidden , out_features).to(device)
    model.load_state_dict(torch.load('model.pth'))

    # Imposta il modello in modalità di valutazione
    model.eval()

        # Fai le previsioni sul set di test
    with torch.no_grad():
        logits = model(graph_test, node_features)
        _, predictions = torch.max(logits, dim=1)
    return predictions

    # plottare bb, linee
    # quando pagina è 0 nuovo documento 
def get_images(type):
    path_json = 'HRDS/' + type +'/'
    path_image= 'HRDS/images/'
    list_j = os.listdir(path_json)
    # prendo il nome dal json 
    image_list = []
    for i in list_j:
        name  = i.replace('.json','')
       # check = os.path.exists(path_json + name)
        #if check:
        image_list.append(name)
    print(len(image_list))
    return path_image, image_list
        

# ad ogni arco assegno quello predict
def plot_graph():
    ''
def main():
    path_image, image_list = get_images('test')
    graph = get_graphs('test')
    graph_test = dgl.batch(graph) # num_nodes=725391, num_edges=811734,
    graph_test = graph_test.int().to(device)

    predictions = inference(graph_test)
    
    print(graph_test)
    print(len(predictions))
    i,j = graph_test.edges()

    # stesso indice !!! 
    edges = list(zip(i.tolist(), j.tolist()))
    page = graph_test.ndata['page'].tolist()
    bb = graph_test.ndata['bb'].tolist()
    centroids = graph_test.ndata['centroids'].tolist() # centroide del nodo i-esimo
    print(len(edges), len(page), len(image_list))

    new_cent = []
    new_bb = []

    check_folder  = True
    if not os.path.exists('savepred/'):
         os.makedirs('savepred')
    
    count = 0 # primo doc
    num_page = 0
    for i in range(len(edges)):
        u, v = edges[i]
        #print(page[u], page[v])
        
        #documento count (es. 10, poi trova 0)
        if i > 0 and page[edges[i-1][1]] > page[v] and page[u] == 0:
            count = count + 1
            check_folder  = os.path.exists(path_image + image_list[count] + '/')
            num_page = 0 # nuovo documento
            new_cent = []
        if check_folder:
            if page[u] == page[v]:  # Se la pagina è la stessa, aggiungi i centroidi e le bb
                new_cent.append(tuple([centroids[u], centroids[v]]))

            elif page[u] == num_page and page[u]!= page[v]:
                # Devi spostare i centroidi di v e aggiungere tutte le altre informazioni
                k = i + 1
                #print('wui')
                if k < len(edges):
                    u1, v1 = edges[k]
                image1, width = get_name(path_image, image_list, page, count, u)
                image2, _ = get_name(path_image, image_list, page, count, v)
                while k < len(edges) and page[u] < page[v1] <= page[u] + 1:
                    if page[u1] != page[v1]:
                        new_cent.append(tuple([[centroids[u1][0], centroids[u1][1]], [centroids[v1][0] +  width, centroids[v1][1]]]))
                    else:
                        new_cent.append(tuple([[centroids[u1][0] +  width, centroids[u1][1]], [centroids[v1][0] +  width, centroids[v1][1]]]))
                    k = k +  1
                    if k < len(edges):
                        u1, v1 = edges[k]
                
                plot_edge(image_list, i, page, new_cent, count, u, v, image1, image2)
                new_cent = []
                num_page =num_page + 1 

def plot_edge(image_list, i, page, new_cent, count, u, v, image1, image2):
    path_save_conc = 'savepred/' + image_list[count]+ '_' + str(page[u]) +'_'+ str(page[v])+'.jpg'
    con_img, con_draw = get_concat_h(image1, image2)#.save(path_save_conc)
    index = i + 1
    for cu, cv in new_cent:
        index = index + 1
        con_draw.line([tuple(cu), tuple(cv)], fill='blue', width=1)
    con_img.save(path_save_conc)# cambiare pagina
 
            
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    draw = D.Draw(dst)
    return dst, draw

def get_name(path_image, image_list, page, count, u):
    name_img = path_image + image_list[count] + '/' + image_list[count] + '_' + str(page[u]) +'.jpg'
    image = Image.open(name_img)
    #draw = D.Draw(image)
    return image, image.width    
''''
una funzione che prende in input 2 immagini, bb, labels, edge, centroid
devo creare centroidi -> della prima immagine rimangono cosi nella seconda shifto
un for per la prima immagine e poi vado avanti con un 
while per inserire tutte le informazioni della seconda immagine
stesso doc :
    page[u] == page[v] -> stessa pagina  
    page[u] != page[v] -> pagina differente
cambio doc : 

'''



if __name__ == '__main__':
    main()