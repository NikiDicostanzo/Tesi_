import csv
import json
import argparse
import os
from PIL import Image, ImageDraw as D
### METTERE UTLIMA PAGINA RIVEDEREE
'''
    {'author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 
    'fig', 'mail', 'secx', 'title', 
    'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}

{meta, contain, connect, equality}

'''

### DOC
# TRAIN: 394 pdf => 8550  %65 -> 1318-
    # TEST:  301 pdf => 4634  %35
    # VAL: fino a doc 62 =>    1331

### HRS
# TRAIN: 900 pdf => 8286  %65 -> 1318-
    # TEST:  100 pdf => 938  %35
    # VAL: fino a doc 62 =>    1331
def create_folder(folder):
    if not os.path.exists(folder):
         os.makedirs(folder)
    
# x0, y0, x1, y1
def create_ann(box, size_w, size_h):
    center = [((box[2] + box[0]) / 2) / size_w, ((box[3] + box[1]) / 2) / size_h]

    w = (box[2] - box[0]) / size_w  # (xmax - xmin)
    h = (box[3] - box[1]) / size_h  # (ymax - ymin)
    #print(center, 'w: ', w, 'h: ', h)
    #print(box,'\n')
    txt_data = str(center[0]) + ' ' + str(center[1]) + ' ' + str(w) + ' ' + str(h) 
    return txt_data 

#OLD dataset
# path_json-> train o val
def get_data(path_json, folder, pagine):
    
    with open(path_json, errors="ignore") as json_file:
        j = json.load(json_file)
        #pagine = 0
        
        
        for m in range(len(j)): #ciclo sui doc-> cioè su 'lines'
            img_pdf = j[m]["imgs_path"]
            lines = j[m]["lines"]
            
            pagine += len(lines)

            # if m < 63 :
            #     folder_dest = folder + 'val/'
            #     print('Doc VAL: ', m, ' Pagine: ', len(lines),'tot: ', pagine)
            # else:
            #     folder_dest = folder + 'train/'
            #     print('Doc TRAIN: ', m, ' Pagine: ', len(lines),'tot: ', pagine)
            folder_dest = folder + 'test/'

            for k in range(len(lines)): # 1 pdf con + immagini
                 
            # print((j[0]['imgs_path'][k])) # immagini che devo rinominare e spostare in Dataset/images/
                
                image = Image.open(img_pdf[k])
                width, height = image.size

                path_name = img_pdf[k].split('/')
                len_name = len(path_name)
                name = path_name[len_name-2] + '_' + path_name[len_name-1].split('.')[0]
                name_image  = name + '.jpg'#'.png'
                name_labels = name + '.txt'

                write = []
                is_title = '2'
                
                for l in lines[k]: # 1 immagine (righe) -> tuti nello stesso txt        
                    # label_idx x_center y_center width height
                    #   label_idx = is_title (0, 1)
                    if l['is_title'] == True:
                        is_title = '0'
                    else:
                        is_title = '1'

                    norm_box = get_norm_box(width, l, height)
                    txt_data = create_ann(norm_box, width, height)

                    tmp = is_title + ' ' + txt_data  
                    write.append(tmp)

                save_lab = folder_dest + 'labels/' + name_labels
                with open(save_lab, 'w') as f:
                    for i in write:
                        f.write(i)
                        f.write('\n')
                save_im = folder_dest + 'images/' + name_image
                image.save(save_im)

#NEW dataset
# path_json-> train o val
def get_data(path_json, path_image, folder, pagine, name,num):
    print(path_json)
    with open(path_json, errors="ignore") as json_file:
        data = json.load(json_file)
        #pagine = 0
        len_data = len(data)
        pagine += data[len_data-1]["page"]
        num+=1
#TODO 
        # if pagine < 100:
        #      folder_dest = folder + 'val/'
        #      print('Doc VAL:', pagine)
        # else:
        #      folder_dest = folder + 'train/'
        print('Doc TRAIN: ', pagine)  
        folder_dest = folder + 'test/' #TODO

        create_folder(folder_dest)
        save_lab_path = folder_dest + 'labels/'
        create_folder(save_lab_path)
        save_im_path  = folder_dest + 'images/'
        create_folder(save_im_path)
        count_page = 0
        write = []
     
        for index in range(len(data)):
            #se cambia pagina salvo
            if index > 0 and data[index]['page'] != data[index-1]['page']: 
                save_lab = save_lab_path + name_labels
                
                with open(save_lab, 'w') as f:
                    for i in write:
                        f.write(i)
                        f.write('\n')
                save_im = save_im_path + name_image
                image.save(save_im) 
                write = []

            page = data[index]['page']
            ############################################################
            if name_dataset == 'HRDS':
                im = path_image + '_' + str(page) + '.jpg'  # TODO per HRDS
            else:
                im = path_image + str(page) + '.png'       # TODO per HRDH
            
            if not os.path.exists(im):
                print('Image: ', im, 'NO EXIST')
                break
            image = Image.open(im)

            width, height = image.size
            name_image  = name +'_' +str(page) + '.png'
            name_labels = name +'_'+ str(page) +'.txt'
            
            """ names

        {'author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 
        'fig', 'mail', 'secx', 'title', 
        'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}
            opara -> diventa quello precedente!
            sec = [title, sec2, sec3, secx...]
            para = [fstline, equ]
            fig = [tab, tabcap, figcap, alg]
            0: sec1
            1: para
            2: fig
            3: meta 
            """     
            #class_lab = ''
            # Classes 
            """ names:
            0: sec1
            1: sec2
            2: sec3
            3: fstline
            4: para
            5: equ
            6: tab
            7: fig
            8: meta 
            9: other
            """  
            # label_idx x_center y_center width height
            #   label_idx = is_title (0, 1)
            
            class_lab = set_class_nine(data, index)
            #class_lab = set_class_four(data, index)
            txt_data = create_ann(data[index]['box'], width, height)
          
            tmp = class_lab + ' ' + txt_data  
            #if class_lab== '0':
            write.append(tmp)
        return pagine,num


def set_class_four(data, index):
    i = 0 #resetto
    name_class = data[index]['class'] 
    if name_class == 'opara':
        i = 1
        # assegno la classe precedente di opara 
        while index - i> 0 and data[index-i]['class'] == 'opara':
            i = i + 1 
            
    if name_class in ['sec1', 'sec2', 'sec3', 'title', 'secx'] \
        or (i > 0 and data[index-i]['class'] in ['sec1', 'sec2', 'sec3', 'title', 'secx'] \
            and name_class == 'opara') :
        class_lab = '0'
    elif name_class in ['fstline', 'para', 'equ'] \
        or (i > 0 and data[index-i]['class'] in ['fstline', 'para', 'equ'] \
            and name_class == 'opara'):
        class_lab = '1'
    elif name_class in ['tab', 'alg', 'fig'] \
        or (i > 0 and data[index-i]['class'] in ['tab', 'alg', 'fig']\
            and name_class == 'opara'):
        class_lab = '2'
    elif data[index]['is_meta'] == True \
        or (i > 0 and data[index-i]['is_meta'] == True  \
            and name_class == 'opara'):
        class_lab = '3'
    elif name_class in ['tabcap', 'figcap'] \
        or (i > 0 and data[index-i]['class'] in ['tabcap', 'figcap']  \
            and name_class == 'opara'):
        class_lab = '4'
    else:
        class_lab='5'
       # print( data[index]['class'])
       # print('perso', name_class)
        
    return class_lab


def set_class_nine(data, index):
    name_class = data[index]['class'] 
    if name_class == 'sec1':
        class_lab = '0'
    elif name_class == 'sec2':
        class_lab = '1'
    elif name_class == 'sec3':
        class_lab = '2'
    elif name_class == 'fstline':
        class_lab = '3'
    elif name_class == 'para':
        class_lab = '4'
    elif name_class == 'equ':
        class_lab = '5'
    elif name_class == 'tab':
        class_lab = '6'
    elif name_class == 'fig':
        class_lab = '7'
    elif data[index]['is_meta'] == True:
        class_lab = '8' 
    else:
        class_lab = '9'
    return class_lab
    
def all_json(path_json, path_images, name_dataset):
    list_json = os.listdir(path_json)

    pagine = 0
    num = 0
    
    for j in list_json:
        if j != '.DS_Store':
            json = path_json + j
            name = j.replace('.json', '')
           
            path_image = path_images + name +'/'
            if name_dataset == 'HRDS':
                image = path_image + name #TODO + name per HRDS !!!!!!!!! per HRDH togli name
            else:
                image = path_image #TODO + name per HRDS !!!!!!!!! per HRDH togli name

            folder = "../dataset_hrdh_all9/"#"C:/Users/ninad/Desktop/Tesi/dataset/"  #dove salvo dati
            pagine,num = get_data(json, image, folder, pagine, name, num)
            
    print('TOTALE', pagine,num)#TOTALE 7788 600

def get_norm_box(width, l, height):
    norm_box = [] 
    for num in l['box']:
        norm = (num * 0.8)#width) / (height)
        norm_box.append(norm)
    return norm_box

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video", dest="video", default=None, help="")
    args = parser.parse_args()
   # path_json = "dataset/train.json"
    name_dataset = 'HRDH'
    path_json = name_dataset+ "/test/" #"HRDS/test/"
    path_images = name_dataset + "/images/"
    print(path_json)
    all_json(path_json, path_images, name_dataset)

    #folder = "C:/Users/ninad/Desktop/Tesi/dataset/"  #dove salvo dati
    #get_data(path_json, folder)
