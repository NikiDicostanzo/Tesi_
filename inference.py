import torch
from plot_edge_multipage_YOLO_new import get_graph_yolo
from model import Model
from create_graphGT import get_graph_merge, get_graphs
import torch.nn.functional as F
from create_graphGT_3lab import get_graphs_gt
import os
import dgl
from PIL import Image, ImageDraw as D
from train import get_nfeatures
from plot_edge_multipage_GT_Merge import get_graph_merge_gt

from evaluation import get_cm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import numpy as np
def get_bb(bb, w, h):
    x0 = bb[0] * w
    y0 = bb[1] * h
    x1 = bb[2] * w
    y1 = bb[3] * h
    return x0, y0, x1, y1

def get_centr(centr, w, h):
    x = centr[0] * w
    y = centr[1] * h
    return x, y

def inference(graph_test, model_name, num_class):
    node_features, input, edge_label = get_nfeatures(graph_test)

    out_features = num_class
    hidden = 20

    # Carica il modello addestrato
    model = Model(input, hidden , out_features).to(device)
    model.load_state_dict(torch.load(model_name))

    # Imposta il modello in modalità di valutazione
    model.eval()

        # Fai le previsioni sul set di test
    with torch.no_grad():
        logits = model(graph_test, node_features)
      #  _, predictions = torch.max(logits, dim=1)


        probabilities = F.softmax(logits, dim=1)
        print(probabilities)
        _, predictions = torch.max(probabilities, dim=1)
       # top_p, top_class = probabilities.topk(3, dim=1)
  
    print('qui',set(np.array(predictions)))
    return predictions

def get_images(type, exp):#TODO CAMBIARE PER YOLO e GT
    if exp == 'yolo':
        path_json = 'zexp_yolo_9_hrdh/json_yolo/'
        path_image= 'zexp_yolo_9_hrdh/savebox/' 
    else:
        path_json = 'HRDS/' + type +'/'
        path_image= 'savebox_no_train/' #'HRDS/images/' 
    list_j = os.listdir(path_json)
    # prendo il nome dal json 

    image_list = []
    for i in (list_j):
        name  = i.replace('.json','')
        image_list.append(name)
    return path_image, (image_list)

def draw_save_edge(folder_save, path_image, image_list, page, graph_test, predictions, exp):
    i_node, j_node = graph_test.edges()
    edges = list(zip(i_node.tolist(), j_node.tolist()))
    bb_norm = graph_test.ndata['bb'].tolist()
    bb = [get_bb(box, 596, 842) for box in bb_norm ]

   # page = graph_test.ndata['page'].tolist()
    
    centroids_norm = graph_test.ndata['centroids'].tolist() # centroide del nodo i-esimo
    centroids =  [get_centr(cent, 596, 842) for cent in centroids_norm ]

    new_cent = []
    new_bb = []
    lab = []
    texts = []
    
    check_folder  = True
    check_path(folder_save)
    count = 0 # primo doc
    num_page = 0
   
    for i in range(len(edges)):
        u, v = edges[i]
        #print(page[u], page[v])
        #documento count (es. 10, poi trova 0)
        #if page[edges[i-1][1]] > page[v]: #page[u] == 0:
        #   print(page[edges[i-1][1]],'|', page[v], page[u] )
        if i > 0 and page[edges[i-1][1]] > page[v] and page[u] == 0:
            count = count + 1
           
            #TODO YOLO
            if exp == 'GT':
                check_folder  = os.path.exists(path_image)
            else: #TODO GT
                check_folder  = os.path.exists(path_image + image_list[count] + '/')

         #   print(count, check_folder, path_image + image_list[count] + '/')

            num_page = 0 # nuovo documento
            new_cent = []
            lab = []
            texts = []
        if check_folder:
            if page[u] == page[v]:  # Se la pagina è la stessa, aggiungi i centroidi e le bb
                new_cent.append(tuple([centroids[u], centroids[v]]))
                lab.append(predictions[i])
               # texts.append([text[u], text[v]]) # solo per controllo -
            elif page[u] == num_page and page[u]!= page[v]:
               # if  image_list[count] == 'ACL_2020.acl-main.99' and page[u] == 2:
               #  print('uuuu', i, '|', text[u], '|', text[v], 'I',len(lab), 'page',page[u], page[v] )
                k = i
                if k < len(edges):
                    u1, v1 = edges[k]
                image1, width = get_name(path_image, image_list, page, count, u, '.png')
                image2, _ = get_name(path_image, image_list, page, count, v, '.png')
               # print(image_list[count], page[u], page[v],  '|',u, v)
                # Plotto gli edge del 2 documento a dx
                while k < len(edges) and page[u] < page[v1] <= page[u] + 1:
                    if page[u1] != page[v1]:
                        new_cent.append(tuple([[centroids[u1][0], centroids[u1][1]], [centroids[v1][0] +  width, centroids[v1][1]]]))
                    else:
                        new_cent.append(tuple([[centroids[u1][0] +  width, centroids[u1][1]], [centroids[v1][0] +  width, centroids[v1][1]]]))
                    lab.append(predictions[k])
                    #texts.append([text[u1], text[v1]])
        
                    k = k +  1
                    if k < len(edges):
                        u1, v1 = edges[k]
                
                plot_edge(texts, image_list, page, new_cent, count, u, v, image1, image2, lab, folder_save)
                new_cent = []
                lab = []
                texts = []
                num_page =num_page + 1

def check_path(folder_save):
    if not os.path.exists(folder_save):
         os.makedirs(folder_save)

def plot_edge(texts, image_list, page, new_cent, count, u, v, image1, image2, lab, folder_save):
    path_save_conc = folder_save + image_list[count]+ '_' + str(page[u]) +'_'+ str(page[v])+'.jpg'
    con_img, con_draw = get_concat_h(image1, image2)
    index = 0
    for cu, cv in new_cent:
        if lab[index] == 1:
            color = 'blue'
            wid = 2
           # con_draw.line([tuple(cu), tuple(cv)], fill=color, width=wid)
        elif lab[index] == 2:
            color = 'cyan'
            wid = 2
           # con_draw.line([tuple(cu), tuple(cv)], fill=color, width=wid)

        elif lab[index] == 0:
            color = 'red'
            wid = 1
   
        con_draw.line([tuple(cu), tuple(cv)], fill=color, width=wid)
        index = index + 1
    con_img.save(path_save_conc)
 
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    draw = D.Draw(dst)
    return dst, draw

def get_name(path_image, image_list, page, count, u, type_im):
    #TODO PER GT
    name_img = path_image + image_list[count] + '/' + image_list[count] + '_' + str(page[u]) + type_im#'.jpg'
    #TODO PER YOLO
    #name_img = path_image + image_list[count] + '_' + str(page[u]) + type_im#'.jpg'
    if os.path.exists(name_img):
        image = Image.open(name_img)
        wid = image.width 
    else:
        image = None
        wid = 0
        #draw = D.Draw(image)
    #draw = D.Draw(image)
    return image, wid   
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

def main(folder_save, model_name, name, exp):
    path_image, image_list = get_images('train', exp) #image_list
   # graph, page, centroids_norm_, image_list =get_graph_merge_gt()#get_graphs('test') 
    if exp == 'yolo':
        graph, page = get_graph_yolo()
    else:
        graph, page, centroids_norm_,_= get_graphs_gt('train') 
   
    graph_test = dgl.batch(graph) # num_nodes=725391, num_edges=811734,
    graph_test = graph_test.int().to(device)
    
    y_true = graph_test.edata['label'] #GT
  #  predictions = inference(graph_test, model_name, 3) # num_class
    
    class_names = [0,1,2]
  #  get_cm(name, np.array(y_true), np.array(predictions),class_names)
   # print(image_list, image_list)
  #  draw_save_edge(folder_save, path_image, image_list, page, graph_test, predictions) 
    draw_save_edge(folder_save, path_image, image_list, page, graph_test, y_true, exp) #


if __name__ == '__main__':
    name = 'bb_lab_cent_rel6_3class'
    folder_save = name + '/'#'sp_bb_lab_rel6_cent/' #'sp_bb_lab_rel_cent_blue/'
    model_name = 'model_' + name + '.pth'#'Pesi/model__bb_lab_rel_cent.pth'
    main(folder_save,model_name, name, 'gt')