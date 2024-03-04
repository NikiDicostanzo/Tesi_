import os
import json
from PIL import Image, ImageDraw as D
from create_graphGT import get_edge_node
def merge_box(box1, box2):
    x0 = min(box1[0], box2[0])
    y0 = min(box1[1], box2[1])
    x1 = max(box1[2], box2[2])
    y1 = max(box1[3], box2[3])
    return x0, y0, x1, y1

def plot_bb(box, draw, color):
    for bb in box:
        draw.rectangle(bb, outline = color) 

def process_json(file_json, folder_save, name, path_images):
    color = 'black'
    folder = path_images + name + '/' 
    draw, image, name_image = get_draw(name, folder, 0) #inizializzo
    with open(file_json, errors="ignore") as json_file:
        data = json.load(json_file)
        bounding_boxes = [item['box'] for item in data] # if item['page'] == page] #Per una pagina!!
        centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in bounding_boxes ]
        page, relation, parent, labels = get_info_json(data)

        labels_edge = []
        array_edges = []
        next_edge = []
        labels_edge_next = []

        x0, y0, x1, y1 = merge_box(data[0]['box'], [10000,10000, 0, 0])  
        for i in range(len(data)): # i = data[i]['line_id']
           # draw.rectangle(data[i]['box'], outline = 'yellow')
            #passa alla nuova pagina, salvo 
           # print(page[i])
            if i> 0 and page[i] != data[i-1]['page']:                 
                draw.rectangle([x0,y0,x1,y1], outline = color)
                plot_edge(draw, centroids, array_edges, labels_edge, 0)
                #if save_image:
                path_save = folder_save  + name_image + '.jpg'
                image.save(path_save)
                    #save_image = False
                # devo aprire immagine vecchia salvata prima se esiste
                if page[i]>1: # inzializzo per la 1 pagina -> ho le prime 2 fatte(una è image)
                    #name è = per tutti
                    draw0, image0, name_image0 = get_draw(name, folder_save, page[i]-2)
                    print(folder_save,  name_image0, '_', name_image)
                    path_save_conc = folder_save +name +'_' + str(page[i]-1) +'_'+ str(page[i])+ '.jpg'
                    get_concat_h(image0, image).save(path_save_conc)
                    
                    draw_con, image_con, name_con = get_draw(name, folder_save, str(page[i]-1) +'_'+ str(page[i]))
                    # creare funzione per disegnare archi tra pagine diverse
                    plot_edge(draw_con, centroids, next_edge, labels_edge_next, image.width) 
                    image_con.save(path_save_conc)
                    next_edge = []
                    labels_edge_next = []

                labels_edge = [] # vedere per piu pagine dove metterlo
                array_edges = []
                draw, image, name_image = get_draw(name, folder, page[i])
            #print(labels) 

            if 'sec' in data[i-1]['class'] or (i>1 and 'sec' in data[i-2]['class'] and 'opara' == data[i-1]['class'] ):
                color = 'red'
            elif 'fig' in data[i-1]['class']:
                color = 'blue'
            elif 'tab' in data[i-1]['class']:
                color = 'green'
            elif 'para' in data[i-1]['class'] or 'equ' in data[i-1]['class']:
                color = 'yellow'
            else:
                color = 'black'

            if i >0 and relation[i]=='connect' and parent[i] == data[i-1]['line_id']:
                if abs(data[i]['box'][0] - x0) < 200: #--300, 30
                    x0, y0, x1, y1 = merge_box(data[i]['box'], [x0, y0, x1, y1])
                else: #se non sono nella stessa colonna
                    draw.rectangle([x0,y0,x1,y1], outline = color) 
                    x0, y0, x1, y1 = merge_box(data[i]['box'], [10000,10000, 0, 0])                
            else:
                if page[i] ==data[i-1]['page']:
                     draw.rectangle([x0,y0,x1,y1], outline = color) 

                x0, y0, x1, y1 = merge_box(data[i]['box'], [10000,10000, 0, 0])
            
            k = 1
            prova = True
            while k< 10 and i - k >0: #and k<i+2 :
                if page[i] == page[i-k]:
                    if i >0 and relation[i]=='connect' and parent[i] == data[i-k]['line_id']:
                        array_edges.append([i-k,i]) # arco con quello precedente
                        labels_edge.append(1)
                    elif k==1 or bounding_boxes[i-k][0] - bounding_boxes[i][0] >150:
                        array_edges.append([i-k,i]) # arco con quello precedente
                        labels_edge.append(0)
                else:
                    if i >0  and relation[i-k] !='meta':# and k>2:
                     #   k=10
                        if i >0 and relation[i]=='connect' and parent[i] == data[i-k]['line_id']:
                            next_edge.append([i-k,i]) # arco con quello precedente
                            labels_edge_next.append(1)
                           # prova = True
                        elif prova == True: #salvo solo 1 rosso
                            next_edge.append([i-k,i])
                            labels_edge_next.append(0)
                            prova = False
                k = k + 1
            
           # if page[i]  != page[i-1]:


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_info_json(data):
    page = [item['page'] for item in data] #data[i]['page']
    labels = [item['class'] for item in data] #data[i]['class']
    relation = [item['relation'] for item in data] # data[i]['relation']
    parent = [item['parent_id'] for item in data] #data[i]['parent_id']
    return page,relation,parent, labels

def plot_edge(draw, point, array_edges, labels_graph, spos):
    index = 0
    for u, v in array_edges:
        if labels_graph[index] == 1:
            color = 'blue'
            wid = 2
        else:
            color = 'red'
            wid = 1
        
        draw.line([point[u], tuple([point[v][0]+spos, point[v][1]])], fill=color, width=wid)
        index = index + 1

def get_draw(name, folder, page):#tutte le informazioni delle immagini stanno nello stesso json
    name_image = name +  '_' + str(page) 
    path_image = folder + name_image + '.jpg'  
    print(path_image)

    image = Image.open(path_image)
    draw = D.Draw(image)
    return draw, image, name_image


def main():
    folder_save = 'gt_plot_testconc/'
    path_jsons = 'HRDS/test/'
    path_images = 'HRDS/images/'

    d = "ACL_2020.acl-main.200.json" #'ACL_2020.acl-main.99.json' #'NAACL_N19-1390.json' #
    name = d.replace('.json', '')

    file_json = os.path.join(path_jsons, d)
    process_json(file_json, folder_save, name, path_images)

if __name__ == '__main__':
    main()

'''
    {'author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 
    'fig', 'mail', 'secx', 'title', 
    'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}

{meta, contain, connect, equality}

'''