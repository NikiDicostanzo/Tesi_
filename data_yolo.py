import csv
import json
import argparse
import os
from PIL import Image, ImageDraw as D

# x0, y0, x1, y1
def create_ann(box, size_w, size_h):
    center = [((box[2] + box[0]) / 2) / size_w, ((box[3] + box[1]) / 2) / size_h]

    w = (box[2] - box[0]) / size_w  # (xmax - xmin)
    h = (box[3] - box[1]) / size_h  # (ymax - ymin)
    #print(center, 'w: ', w, 'h: ', h)
    #print(box,'\n')
    txt_data = str(center[0]) + ' ' + str(center[1]) + ' ' + str(w) + ' ' + str(h) 
    return txt_data 



# path_json-> train o val
def get_data(path_json, folder):
    # TRAIN: 394 pdf => 8550  %65 -> 1318-
    # TEST:  301 pdf => 4634  %35
    # VAL: fino a doc 62 =>    1331
    with open(path_json, errors="ignore") as json_file:
        j = json.load(json_file)
        pagine = 0
        
        for m in range(len(j)): #ciclo sui doc-> cioè su 'lines'
            img_pdf = j[m]["imgs_path"]
            lines = j[m]["lines"]
            
            pagine += len(lines)
            if m < 63 :
                folder_dest = folder + 'val/'
                print('Doc VAL: ', m, ' Pagine: ', len(lines),'tot: ', pagine)
            else:
                folder_dest = folder + 'train/'
                print('Doc TRAIN: ', m, ' Pagine: ', len(lines),'tot: ', pagine)

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
    path_json = "dataset/train.json"
    folder = "C:/Users/ninad/Desktop/Tesi/dataset/"  #dove salvo dati
    get_data(path_json, folder)
