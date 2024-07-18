import torch
from plot_edge_multipage_YOLO_new import get_graph_yolo
from plot_edge_multipage_PARSE import get_graph_parse
#from plot_edge_multipage_YOLO5 import get_graph_yolo

from model import Model
from create_graphGT import get_graph_merge, get_graphs
import torch.nn.functional as F
from create_graphGT_3lab import get_graphs_gt
import os
import dgl
from PIL import Image, ImageDraw as D
from train import accuracy, get_nfeatures
from z_boh.plot_edge_multipage_GT_Merge import get_graph_merge_gt

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

def inference(graph_test, model_name, num_class, array_features):
    node_features, input, edge_label = get_nfeatures(graph_test, array_features)

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
    
    acc = accuracy(logits, edge_label)

    print('Test Accuracy: {:.4f}'.format(acc)) # Test Accuracy
   # print('qui',set(np.array(predictions)))
    return predictions

def get_images(type, exp):#TODO CAMBIARE PER YOLO e GT
    if exp == 'yolo':
        path_json = 'yolo_hrdhs_672_9/json_yolo/' #bbGT_labYolo/'# 
        path_image= 'yolo_hrdhs_672_9/savebox/' 
    elif exp == 'parse':
        print('PARSE')
        path_json = 'yolo_hrds_4_gt_test/check_json_label/' #bbGT_labYolo/'# 
        path_image= 'yolo_hrds_4_gt_test/plot_bb_parse/' 
    else:
        path_json = 'HRDS/' + type +'/'
        path_image= 'savebox_test/'#'HRDS/images/' #'savebox_'+ type+ '/' #
    list_j = os.listdir(path_json)
    # prendo il nome dal json 

    image_list = []
    for i in (list_j):
        name  = i.replace('.json','')
        image_list.append(name)
    return path_image, (image_list)

def draw_save_edge(folder_save, path_image, image_list, page, graph_test, predictions, exp, name_exp):
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
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
       # print(page[edges[i-1][1]],'|', page[v], page[u] )
        if i > 0 and page[edges[i-1][1]] > page[v] and page[u] == 0:
            count = count + 1
           
            #TODO YOLO
            if exp == 'yolo' or exp=='parse':
                check_folder  = os.path.exists(path_image)
            else: #TODO GT
                check_folder  = os.path.exists(path_image + image_list[count] + '/')
            #print(count, check_folder, path_image + image_list[count] + '/')

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
                #print(path_image, image_list[count], page[u])
                image1, width = get_name(path_image, image_list, page, count, u, exp)
                image2, _ = get_name(path_image, image_list, page, count, v, exp)
                if image2 == None: # per ultimo 
                    break
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
                
                plot_edge(texts, image_list, page, new_cent, count, u, v, image1, image2, lab, folder_save,name_exp)
                new_cent = []
                lab = []
                texts = []
                num_page =num_page + 1

def check_path(folder_save):
    if not os.path.exists(folder_save):
         os.makedirs(folder_save)

def plot_edge(texts, image_list, page, new_cent, count, u, v, image1, image2, lab, folder_save, name_exp):
    path_save_conc = folder_save + image_list[count]+ '_' + str(page[u]) +'_'+ str(page[v])+'_'+name_exp+'.jpg'
    con_img, con_draw = get_concat_h(image1, image2)
    index = 0
    for cu, cv in new_cent:
        if lab[index] == 1:
            color = 'blue'
            wid = 2
            #con_draw.line([tuple(cu), tuple(cv)], fill=color, width=wid)
        elif lab[index] == 2:
            color = 'cyan'
            wid = 2
           # con_draw.line([tuple(cu), tuple(cv)], fill=color, width=wid)
#
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

def get_name(path_image, image_list, page, count, u, exp):
   
    #TODO PER YOLO
    if exp == 'yolo' or exp=='parse':
        name_img = path_image + image_list[count] + '_' + str(page[u]) + '.png'
    else: #TODO PER GT
        name_img = path_image + image_list[count] + '/' + image_list[count] + '_' + str(page[u]) + '.jpg'

    if os.path.exists(name_img):
        image = Image.open(name_img)
        wid = image.width 
    else:
        image = None
        wid = 0
        #draw = D.Draw(image)
    #draw = D.Draw(image)
    return image, wid   

def get_lab_different_page(y_true, predictions, page, edges):
    new_y_true = []
    new_pred = []
    for i in range(len(edges)):
        u, v = edges[i]
        if page[u] != page[v]: # trovo solo gli archi che stanno in pagine diverse
            new_y_true.append(y_true[i])
            new_pred.append(predictions[i])
    return new_y_true, new_pred



def main(name_pth, folder_save, type_data, model_name, name, exp, name_exp, kr, num_class, num_arch_node, num_arch_node2, class3, array_features):
    path_image, image_list = get_images(type_data, exp) #image_list
    print(path_image)
   # graph, page, centroids_norm_, image_list =get_graph_merge_gt()#get_graphs('test') 
    
    if exp == 'yolo':
        graph, page = get_graph_yolo(kr, num_arch_node, class3, type_data, array_features)
    elif exp == 'parse':
        graph, page = get_graph_parse(kr, num_arch_node, num_arch_node2, class3, type_data, array_features)    
   # else:
   #     graph, page, centroids_norm_,_= get_graphs_gt(type_data, kr, num_arch_node,class3, array_features) 
    dimensione_desiderata =11
    dimensione_desiderata_font = 6
     #TODO numero classi
    
    for g in graph:
        print('QUI', g.ndata['labels'].shape[1])
      # Assicurati che tutte le feature dei nodi abbiano la stessa dimensione e tipo di dati
        if 'labels' not in g.ndata or g.ndata['labels'].shape[1] != dimensione_desiderata:
            # Crea una feature di placeholder con la dimensione desiderata
            placeholder = torch.zeros((g.number_of_nodes(), dimensione_desiderata), dtype=torch.float64)
            g.ndata['labels'] = placeholder
        if 'font' not in g.ndata or g.ndata['font'].shape[1] != dimensione_desiderata_font:
            # Crea una feature di placeholder con la dimensione desiderata
            placeholder = torch.zeros((g.number_of_nodes(), dimensione_desiderata_font), dtype=torch.float64)
            g.ndata['font'] = placeholder

    graph_test = dgl.batch(graph) # num_nodes=725391, num_edges=811734,
    graph_test = graph_test.int().to(device)
    
    y_true = graph_test.edata['label'] #GT
    predictions = inference(graph_test, model_name, num_class, array_features) # num_class
   # print(set(predictions))
    if class3:
        class_names = [0,1, 2] 
    else:
        class_names = [0,1]
    
    i_node, j_node = graph_test.edges()
    edges = list(zip(i_node.tolist(), j_node.tolist()))
    new_y_true, new_predictions = get_lab_different_page(y_true.cpu().numpy(), predictions.cpu().numpy(), page, edges)

    get_cm(name, name_pth + '_all_pred', y_true.cpu().numpy(), predictions.cpu().numpy(),class_names)
    get_cm(name, name_pth + '_page_diff_pred', new_y_true, (new_predictions),class_names)
   
   # get_cm(name, name_pth + '_all_pred', np.array(y_true), np.array(predictions),class_names)
   # get_cm(name, name_pth + '_page_diff_pred', np.array(new_y_true), np.array(new_predictions),class_names)
    print('Draw')
   # draw_save_edge(folder_save, path_image, image_list, page, graph_test, predictions, exp, name_exp) 
   # draw_save_edge(folder_save, path_image, image_list, page, graph_test, y_true, exp, name_exp) #


if __name__ == '__main__':
    top = 'yolo_hrds_4_gt_test'
    # Cartella dove salvo immagini 
    name_pth ='bb_area_w_h_cent_lab_rel3_agg_pageRel_in1_out1' #bb_area_w_h_cent_lab_rel6_agg_pageRel_in1_out1'#'bb_area_w_h_cent_in1_out1' #'bb_cent_area_w_h_lab_k6_0.5' 

    folder_save = 'yolo_hrds_4_gt_test/savepred/' + name_pth +'/'
 
    #folder_save = name + '/'#'sp_bb_lab_rel6_cent/' #'sp_RRbb_lab_rel_cent_blue/'
    array_features = ['bb', 'cent', 'area', 'h', 'w', 'lab', 'rel', 'agg',  'pageRel']#,'agg', 'rel']

    name_exp = ''#'exp_rel5_lab_agg'  #TODO
    model_name = 'z_check_FIN/model_' + name_pth + '.pth'#'Pesi/model__bb_lab_rel_cent.pth'
    main(name_pth, folder_save, 'test', model_name, top, 'parse', name_exp, 3, 3, 1, 1, True, array_features) # type, kr, num_class, num_arch_node
       #                     kr, num_class, num_arch_node, class3
    print("EXP: ", name_pth)

       #