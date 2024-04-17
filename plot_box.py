import os
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw

from plot_edge_multipage_YOLO_new import get_color
from create_graphGT_3lab import get_info_json

def check_path(folder_save):
    if not os.path.exists(folder_save):
         os.makedirs(folder_save)

def get_name(path_image, name, page):
    #name_img = path_image + name + '_' + str(page[u]) +'.png'
    name_img = path_image + name + '/' + name +'_' + str(page) +'.jpg'
    print(os.path.exists(name_img), name_img)
    if os.path.exists(name_img):
        image = Image.open(name_img)
        wid = image.width 
    else:
        image = None
        wid = 0
        #draw = D.Draw(image)
  #      print(name_img)
    return image, wid

def plot_box(path_image, path_new_im, bounding_boxes, page, labels, name_image):
    image,_ = get_name(path_image, name_image, 0)
    print('qui',image, name_image)

    draw = ImageDraw.Draw(image)
    for b in range(len(bounding_boxes)):
        p = page[b]
        color = get_color(labels[b])
        check_path(path_new_im + name_image )

        if b>0 and p != page[b-1]: #cambia pagina #salvo
            save_im = path_new_im + name_image + '/' + name_image + '_' + str(page[b-1]) + '.jpg'#'.png'
            
            image.save(save_im)
            image, _ = get_name(path_image, name_image, page[b])
            if image == None:
                break
            draw = ImageDraw.Draw(image)
        elif b == len(bounding_boxes)-1:
            draw.rectangle(bounding_boxes[b], outline = color) 

            save_im = path_new_im + name_image + '/' +name_image + '_' + str(page[b]) + '.jpg'#'.png'
            image.save(save_im)
        draw.rectangle(bounding_boxes[b], outline = color)

def main():
    path = 'HRDS/test/'
    path_new_im = 'savebox_test/'
    path_image = 'HRDS/images/'
    list_json = os.listdir(path)
    for j in list_json:
        json_name = path + j
        name_image = j.replace('.json', '')
        #path_image = 'HRDS/images/'# + name_image
        print(name_image)
        with open(json_name) as f:
            data = json.load(f)
            bounding_boxes, page,relation,parent, labels, text = get_info_json(data)
            plot_box(path_image, path_new_im, bounding_boxes, page, labels, name_image)



if __name__ == '__main__':
    main()