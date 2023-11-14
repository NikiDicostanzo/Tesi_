import csv
import json
import argparse
import os
from PIL import Image, ImageDraw as D
import numpy as np
import shutil

## Disegna bb con GT

def analisi_dati(path_json):
    
    with open(path_json, errors="ignore") as json_file:
        j = json.load(json_file)
        
        for m in range(len(j)): #ciclo sui doc
            img_pdf = j[m]["imgs_path"]
            lines = j[m]["lines"]
           
            im = img_pdf[0].split("/")  #prima pagina del doc
            path = im[0] + "/" + im[1] + "/" + im[2] + "/" + im[3] + "/"
            
            num_page = 0
            print('Title_arch,  Parent,  Relation,     Content')
            relation = []
            parent = []
            if sib_true(lines):
                new_path = 'result2/' #+ im[3] + '/'
                filename = new_path + '/title2.csv'
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                
                with open(filename, 'w', encoding="utf-8") as csvfile:
                    #filewriter = csv.writer(csvfile)
                    #filewriter.writerow(["line_id",'title_arch', 'parent', 'relation', 'content'])
                    ##for i in lines:  #pagine
                        path_img = path + str(num_page) + ".png"
                        i = lines[0]
                        image = Image.open(path_img)
                        draw = D.Draw(image)
                        width, height = image.size
                        # # [x0, y0, x1, y1]
                        
                        for k in i:    #Contenuto
                            norm_box = []
                            if k['relation'] not in relation:
                                relation.append(k['relation'])
                            if k['parent'] not in parent:
                                parent.append(k['parent'])
                            for num in k['box']:
                                norm_box.append(num * 0.8)
                            
                            if k["is_title"] == True: #"relation": "equal",
                                print(k['title_arch'], '       ', k['parent'], '  ', k['relation'], '  ', k["content"])
                                dict_title =[k["line_id"],k['title_arch'], k['parent'], k['relation'], k["content"]]
                                #filewriter.writerow(dict_title)
                            # else:
                            #     color = 'blue'  #"relation": "-1",
                            if k["relation"] == 'contain': #"relation": "equal",
                                color = 'red'
                                #print(k['title_arch'],'\n', k["content"])
                            elif k["relation"] == 'equal':
                                color = 'blue'  #"relation": "-1",
                            else:
                                color = 'black'
                            draw.rectangle(norm_box, outline = color)  # disegno tutti i rettangoli del documento
                        #image.show()
                    
                        #path_save = new_path + str(num_page) + ".png"
                        path_save = new_path + im[3] +'_'+str(num_page) + ".png"
                        # da rivedere dove salvare, (fare cartella per ogni lingua/train-val)

                        image.save(path_save)
                        num_page+= 1

                print("Relation:  " ,relation)
                print("Parent", parent)

def sib_true(lines):
    for i in lines:    
        for k in i:
            if k['relation'] == 'sibling':
                return True
    return False    

# cerca indice per Disegnare bb su un'immagine data
def find_image(j,name):
    for i in range(len(j)):
            #print(name)
            if name in j[i]['imgs_path']:
                #print('Name:', name, ' count ', i)
                return i
    return print('Image not found')
    

# image scale, check if bb scale are corrected 
def draw_box_rid(image, j, index, page):
     # Immagine ridimensionata
        image_rid = Image.open('33.png')
        draw_rid = D.Draw(image_rid)
        width_rid, height_rid = image_rid.size 

        width, height = image.size 

        # [x0, y0, x1, y1]
        #print(width_rid, height_rid)      
        for k in j[index]["lines"][page]:    #Contenuto
            norm_box = []
            c = 0
            for num in k['box']:
                if c == 0 or c == 2:
                    norm_box.append((num * 0.8)/width*width_rid) 
                else: 
                    norm_box.append((num * 0.8)/height*height_rid)
                c += 1

            if k["relation"] == 'contain': #"relation": "equal",
                color = 'red'
                #print(k['title_arch'],'\n', k["content"])
            elif k["relation"] == 'equal':
                color = 'blue'  #"relation": "-1",
            else:
                color = 'black'
            draw_rid.rectangle(norm_box, outline = color)  # disegno tutti i rettangoli del documento
        image_rid.show()

def draw_box(image, j, index, page, save_path):

    draw = D.Draw(image)
    width, height = image.size 
    #477 674 ->h-80= 594 -- w / h -> 0.803
    #490 634 -> h-20..  w / h -> 0.798 

    # [x0, y0, x1, y1]
    all_bb = []
    labels = []
    for k in j[index]["lines"][page]:    #Contenuto
        norm_box = []
        for num in k['box']:
            norm_box.append(num * 0.8)
       
        if k["is_title"] == True: #"relation": "equal",
            color = 'red'
            labels.append(0)
            #print(k['title_arch'],'\n', k["content"])
        else:
            color = 'black'
            labels.append(1)
        draw.rectangle(norm_box, outline = color)  # disegno tutti i rettangoli del documento
        all_bb.append(norm_box)
    #print(len(labels), len(all_bb))
    if save_path != '':
        image.save(save_path)
    # else:
    #    image.show()
    return all_bb, labels


# Disegna bb su un'immagine data
def get_bb_gt(name,path_json, page, path_save):
    
    with open(path_json, errors="ignore") as json_file:
        j = json.load(json_file)
        index = find_image(j , name)
        #print(index)
        image = Image.open(name)
        #draw_box_rid(image, j, index, page)
        all_bb, labels = draw_box(image, j, index, page, path_save)
    return all_bb, labels


def create_folder(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def get_name(d):
    nm = d.split('.')
    vis = nm[1].split('_')
    page  = vis[len(vis)-1]
    image_name= nm[0] +'.' + vis[0] + '_vis'
    path_name = "./dataset/imgs/"  + image_name + '/' + page + '.png'
    return page, path_name, image_name 

#per test_exp
def all_page(folder, path_images, path_json):
    dir_list = os.listdir(path_images) 
    with open(path_json, errors="ignore") as json_file:
        j = json.load(json_file)
        #trovo indice
        path_save = folder + 'gt/'
        create_folder(path_save)
        #es. json -> ./dataset/imgs/astro-ph_16_1608.06702_vis/0.png
        #ciclo su tutte le pagine 
        #page = 0
        for d in dir_list:
            #print(dir_list[0])
            path_image = path_images + d
            save_image= path_save + str(d) + '.png'
            
            #name_json
            page, path_name, image_name  = get_name(d)

            index = find_image(j , path_name)
           # print(path_image)
           # print('index: ', index)
            image = Image.open(path_image)
            #draw_box_rid(image, j, index, page)
            #print('image: ', path_image)
            if save_image != '':
                draw_box(image, j, index, int(page) , save_image)
            #page +=1
    

""" def all_page(folder, path_images, path_json):
    dir_list = os.listdir(path_images) 
    with open(path_json, errors="ignore") as json_file:
        j = json.load(json_file)
        #trovo indice
        name = path_images + '0.png'
        index = find_image(j , name)
        print(name)
        print('index: ', index)
        path_save = folder + '/gt/'

        #ciclo su tutte le pagine 
        page = 0
        for d in range(len(dir_list)):
            #print(dir_list[0])
            path_image = path_images + str(d) + '.png'
            save_image= path_save + str(d) + '.png'
            image = Image.open(path_image)
            #draw_box_rid(image, j, index, page)
            #print('image: ', path_image)
            draw_box(image, j, index, page , save_image)
            page +=1 """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")     path_json = "dataset/train.json"
    path_dataset= './dataset/'
    #path_images = path_dataset + 'imgs/' + name
    folder = 'exp_test/'
    path_images = folder + "images/"
    path_json = path_dataset + "train.json"
    #all_page(folder, path_images, path_json)

    """  
    args = parser.parse_args()
    #analisi_dati(path_json)
    #name = './dataset/imgs/hep-ex_18_1812.10790_vis/' + str(page) + '.png'
    name = './dataset/imgs/q-bio_16_1601.00716_vis/' + str(page) + '.png' """
    page = 13
    name = './dataset/imgs/stat_21_2109.00464_vis/' + str(page) + '.png'
    bb_gt, labels_gt = get_bb_gt(name, path_json, page, '') 



 

    