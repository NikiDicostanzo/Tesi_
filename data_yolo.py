import csv
import json
import argparse
import os
from PIL import Image, ImageDraw as D

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
                name_image  = name + '.png'
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
    
    with open(path_json, errors="ignore") as json_file:
        data = json.load(json_file)
        #pagine = 0
        len_data = len(data)
        pagine += data[len_data-1]["page"]
        num+=1
        #print(pagine)
        #print(num)
        if pagine < 100:
             folder_dest = folder + 'val/'
             print('Doc VAL:', pagine)
        else:
            folder_dest = folder + 'train/'
            print('Doc TRAIN: ', pagine)  
       # folder_dest = folder + 'test/'
        create_folder(folder_dest)
        save_lab_path = folder_dest + 'labels/'
        create_folder(save_lab_path)
        save_im_path  = folder_dest + 'images/'
        create_folder(save_im_path)

        count_page = 0
        write = []
        for index in range(len(data)) :

            #se cambia pagina salvo
            if index > 0 and data[index]['page'] != data[index-1]['page']: 
                
                save_lab = save_lab_path + name_labels
                
               # print(save_lab)
                with open(save_lab, 'w') as f:
                    for i in write:
                        f.write(i)
                        f.write('\n')
                save_im = save_im_path + name_image
                image.save(save_im) 
                write = []

            page = data[index]['page']
            im = path_image + '_' + str(page) + '.jpg'
            image = Image.open(im)
            width, height = image.size
            name_image  = name +'_' +str(page) + '.png'
            name_labels = name +'_'+ str(page) +'.txt'

                
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
            
            txt_data = create_ann(data[index]['box'], width, height)
          
            tmp = class_lab + ' ' + txt_data  
            #if class_lab== '0':
            #      print(tmp, name_labels)
            write.append(tmp)

            
        return pagine,num
    
def all_json(path_json, path_images):
    list_json = os.listdir(path_json)
    pagine = 0
    num = 0
    print(len(list_json))
    
    for j in list_json:
        json = path_json + j
        #print(j)
        name = j.replace('.json', '')
        path_image = path_images + name +'/'
        image = path_image + name 
        #print(path_images)
        folder = "../data_yolo_new/"#"C:/Users/ninad/Desktop/Tesi/dataset/"  #dove salvo dati
        pagine,num = get_data(json, image, folder, pagine, name,num)
        
    print('TOTALE', pagine,num)

def get_norm_box(width, l, height):
    norm_box = [] 
    for num in l['box']:
        norm = (num * 0.8)#width) / (height)
        norm_box.append(norm)
    return norm_box

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create event frame")
    parser.add_argument("--video", dest="video", default=None, help="Path of the video")
    args = parser.parse_args()
   # path_json = "dataset/train.json"
    path_json = "HRDS/train/"
    path_images = "HRDS/images/"
    print(path_json)
    all_json(path_json, path_images)
    #folder = "C:/Users/ninad/Desktop/Tesi/dataset/"  #dove salvo dati
    #get_data(path_json, folder)
