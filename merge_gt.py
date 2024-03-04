import os
import json
from PIL import Image, ImageDraw as D
from plot_edge_multipage_YOLO import get_color


def merge_box(box1, box2):
    x0 = min(box1[0], box2[0])
    y0 = min(box1[1], box2[1])
    x1 = max(box1[2], box2[2])
    y1 = max(box1[3], box2[3])
    return x0, y0, x1, y1

def process_json(file_json, folder_save, name, path_images):
 
    folder = path_images + name + '/' 
    draw, image, name_image = get_draw(name, folder, 0) #inizializzo
    set_lab = False
    with open(file_json, errors="ignore") as json_file:
        data = json.load(json_file)
        box_new = []
        lab_new = []
        page_new =[]
        x0, y0, x1, y1 = merge_box(data[0]['box'], [10000,10000, 0, 0])
        current_lab = 'other'
        for i in range(len(data)): # i = data[i]['line_id']
            page, relation, parent, lab = get_info_json(data, i)

            #passa alla nuova pagina, salvo 
            if i> 0 and page != data[i-1]['page']:  
                draw.rectangle([x0,y0,x1,y1], outline = 'red') 
                box_new.append([x0, y0, x1, y1])            #TODO Add
                lab_new.append(data[i-1]['class'])
                page_new.append(data[i-1]['page'])
                path_save = folder_save  + name_image 
                image.save(path_save)
                draw, image, name_image = get_draw(name, folder, page)
            
            if i >0 and relation=='connect' and parent == data[i-1]['line_id']: 
       
                if abs(data[i]['box'][0] - x0) < 200: #--300, 30
                    x0, y0, x1, y1 = merge_box(data[i]['box'], [x0, y0, x1, y1])
                    current_lab = lab
                    set_lab = False # 
                else: #se non sono nella stessa colonna
                    draw.rectangle([x0,y0,x1,y1], outline = 'magenta') 
                    box_new.append([x0, y0, x1, y1])          #TODO Add
                    lab_new.append(current_lab)
                    page_new.append(page)
                    x0, y0, x1, y1 = merge_box(data[i]['box'], [10000,10000, 0, 0])
                  
            else:
                if page ==data[i-1]['page']:
                     draw.rectangle([x0,y0,x1,y1], outline = 'yellow') 
                     box_new.append([x0, y0, x1, y1])            #TODO Add
                     lab_new.append(data[i-1]['class'])
                     page_new.append(data[i-1]['page'])
                x0, y0, x1, y1 = merge_box(data[i]['box'], [10000,10000, 0, 0])
              #  lab_new.append(lab)
    print(len(lab_new),len(box_new))


    image2 = Image.open('HRDS/images/NAACL_N19-1390/NAACL_N19-1390_0.jpg')
    draw2 = D.Draw(image2)
    for i in range(len(box_new)):
        
        if i > 0 and page_new[i] != page_new[i-1]:
            image2.save('plot_merge_gt_lab/NAACL_N19-1390_'+str(page_new[i-1])+'.png')
            image2 = Image.open('HRDS/images/NAACL_N19-1390/NAACL_N19-1390_' + str(page_new[i]) +'.jpg')
            draw2 = D.Draw(image2)
        color = get_color(lab_new[i])
        #if page_new[i] == 0:
        draw2.rectangle(box_new[i], outline = color)




def get_info_json(data, i):
    page = data[i]['page']
    labels = data[i]['class']
    relation = data[i]['relation']
    parent = data[i]['parent_id']
    return page,relation,parent, labels


def get_draw(name, folder, page):#tutte le informazioni delle immagini stanno nello stesso json
    name_image = name +  '_' + str(page) + '.jpg'
    path_image = folder + name_image
    image = Image.open(path_image)
    draw = D.Draw(image)
    return draw, image, name_image


def main():
    folder_save = 'merge_new_gt/'
    path_jsons = 'HRDS/test/'
    path_images = 'HRDS/images/'


    d = 'NAACL_N19-1390.json' #'ACL_2020.acl-main.99.json'
    name = d.replace('.json', '')

    file_json = os.path.join(path_jsons, d)
    process_json(file_json, folder_save, name, path_images)

if __name__ == '__main__':
    main()

'''
    {'author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 'fig', 'mail', 'secx', 'title', 
    'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}
'''