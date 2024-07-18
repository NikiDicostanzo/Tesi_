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

#NEW dataset
# path_json-> train o val
def get_data(path_json, path_image, folder, pagine, name,num,data_type):
    print(path_json)
    with open(path_json, errors="ignore") as json_file:
        data = json.load(json_file)
        #pagine = 0
        len_data = len(data)
        pagine += data[len_data-1]["page"]
        num+=1
        if data_type == 'train': #TODO  
            if pagine < 100:
                folder_dest = folder + 'val/'
                print('Doc VAL:', pagine)
            else:
                folder_dest = folder + 'train/'
            print('Doc TRAIN: ', pagine)  
        else:
            folder_dest = folder + 'test/' #TODO
            print('Doc TEST ', pagine)  

        create_folder(folder_dest)
        save_lab_path = folder_dest + 'labels/'
        create_folder(save_lab_path)
        save_im_path  = folder_dest + 'images/'
        create_folder(save_im_path)
        count_page = 0
        write = []
        save_lab_array = []
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
              #  print('Image: ', im, 'NO EXIST')
                break
            image = Image.open(im)

            width, height = image.size
            name_image  = name +'_' +str(page) + '.png'
            name_labels = name +'_'+ str(page) +'.txt'
            
            if index == len(data)-1:
                save_lab = save_lab_path + name_labels
                with open(save_lab, 'w') as f:
                    for i in write:
                        f.write(i)
                        f.write('\n')
                
                save_im = save_im_path + name_image
                image.save(save_im) 
            if data[index]['class'] in ['tab', 'fig', 'alg', 'equ']:
                class_lab = set_class_3(data, index)#, save_lab_array)#set_class_all(data, index)
                save_lab_array.append(class_lab) # mi serve per opara
                #set_class_nine(data, index)
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

'''
    {'author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 
    'fig', 'mail', 'secx', 'title', 
    'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}


#  0: title
  1: author
  2: mail
  3: affili
  4: sec
  5: fstline
  6: para
  7: tab
  8: fig
  9: cap
  10: equ 
  11: foot
  12: head
  13: fnote
  '''
def set_class_3(data, index):
    
    name_class = data[index]['class'] 
    if name_class == 'tab':
        class_lab = '0'
    elif name_class == 'fig':
        class_lab = '1'
    elif name_class == 'alg':
        class_lab = '2'
    elif name_class == 'equ':
        class_lab = '3'
    return class_lab


'''
    {'author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 
    'fig', 'mail', 'secx', 'title', 
    'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}
'''
#class_names = ['title', 'sec', 'meta', 'caption' , 'para', 'note', 'equ', 'tab', 'alg', 'fig', 'page']
def set_class_nine(data, index, lab):
    name_class = data[index]['class'] 

    if name_class == 'title':
        class_lab = '0'
    elif name_class in ['sec1','sec2', 'sec3', 'secx'] :
        class_lab = '1'
    elif name_class in ['tabcap', 'figcap']:
        class_lab = '2'
    elif name_class in ['para', 'fstline']:
        class_lab = '4'
    elif name_class == 'fnote':
        class_lab = '5'
    elif name_class == 'equ':
        class_lab = '6'
    elif name_class == 'tab':
        class_lab = '7'
    elif name_class == 'alg':
        class_lab = '8'
    elif name_class == 'fig':
        class_lab = '9'
    elif data[index]['is_meta'] == True:
        class_lab = '3' 
    elif len(lab) > 1 and name_class == 'opara':
        class_lab = lab[-1]
    else:
        class_lab = '10'
        print(name_class)
    return class_lab

# class_all = ['author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 
#     'fig', 'mail', 'secx', 'title', 
#     'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili', 'header']
class_all = ['tab', 'fig', 'alg', 'equ']

def set_class_all(data, index):
    name_class = data[index]['class'] 
    dizionario = dict(enumerate(class_all))

# Stampa del dizionario
   # print(name_class)
    class_lab = str([k for k, v in dizionario.items() if v == name_class][0])
  
    return class_lab
    
def all_json(path_json, path_images, name_dataset, data_type):
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

            folder = "../dataset_hrdhx2_TabIMclass/"#"C:/Users/ninad/Desktop/Tesi/dataset/"  #dove salvo dati
            create_folder(folder)
            pagine,num = get_data(json, image, folder, pagine, name, num, data_type)
            
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
    data_type = 'test'
    name_dataset = 'HRDH'
    path_json = name_dataset+ "/"+ data_type + "/" #"HRDS/test/"
    path_images = name_dataset + "/images/"
    print(path_json)
    all_json(path_json, path_images, name_dataset, data_type)

    #folder = "C:/Users/ninad/Desktop/Tesi/dataset/"  #dove salvo dati
    #get_data(path_json, folder)
