import os
import json
from PIL import Image, ImageDraw

from pdfParser import bb_scale, dimension_pdf

def get_info_json(data):
    bounding_boxes = [item['box'] for item in data] # salvo tutte le bb delle miei pagine
    page = [item['page'] for item in data]  
    size = [item['size'] for item in data] 
    text = [item['text'] for item in data]
    type = [item['type'] for item in data]
    style = [item['style'] for item in data]
    font = [item['font'] for item in data]
    return bounding_boxes, page, text, size, type, style, font

#dict_im = {'box': im['bbox'], 'height': im['height'], 'width': im['width'], 'page': k, 'type': 'img'}
#dict_data = {'box' : bb[i], 'style': f_style[i], 'size': f_size[i], 'font': font[i], 'text': text[i], 'page': k, 'type': 'text'}

def get_graph():

    ''
# {'author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 
#    'fig', 'mail', 'secx', 'title', 
#    'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}

# labels = ['author', 'alg', 'sec', 'equ', 'tabcap', 'foot', 'tab', 
#    'fig', 'mail', 'secx', 'title', 
#    'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}]

#['title','sec', 'autho,', 'meta', 'para', 'fnote', 'page', 'other']
def get_color(lab):
    lab = int(lab)
    if lab == 0:
        color = 'red'
    elif lab == 1:
        color = 'cyan'
    elif lab == 2:
        color = 'green'
    elif lab == 3:
        color = 'gray'
    elif lab == 4:
        color = 'yellow'
    elif lab == 5:
        color = 'purple'
    elif lab == 6:
        color = 'blue'
    elif lab == 7:
        color = 'orange'
    elif lab == 9:
        color = 'magenta'
    elif lab == 10:
        color = 'pink'
    elif lab == 11:
        color = '#ECEBBD'
    else:
        color = 'black'

    return color


def draw_all(path, bb, lab, page, name_im, pdf_path):
    #name = path + name_im + '/' + name_im + '_0.png' # iniz.
    name = path + name_im + '_0.png' # iniz.

    image = Image.open(name)
    draw = ImageDraw.Draw(image)
    w , h = image.size
    wp, hp = dimension_pdf(pdf_path)
    print('qqq' , image.size)

    for i in range(len(bb)):
        
        
        if i >0 and page[i] != page[i-1] :
            #path_save = 'bb_check/' + name_im + '_' +str(data[i-1]['page']) + '.jpg'
            path_save = save_path + name_im + '_' +str(page[i-1]) + '.png'

            image.save(path_save)
            #name = path + name_im + '/' + name_im + '_' + str(page[i]) + '.png'
            name = path  + name_im + '_' + str(page[i]) + '.png'
            image = Image.open(name)
            draw = ImageDraw.Draw(image)
        #print(lab[i])
        color = get_color(lab[i])
        draw.rectangle(bb_scale(bb[i], w, h, float(wp), float(hp)), outline = color, width = 2) 
        #image.save('zzz/page0_bb'+ str(i) +'.jpg')
        

def is_number(input_str):
    return input_str.isdigit()

def is_caption(bb, text, i, is_cap):
    if (text[i].split(' ')[0] == 'Figure' or text[i].split(' ')[0] == 'Table')\
          and i >0 and bb[i][1]-bb[i-1][3]<-200: #l'immagine o tabella non stanno accanto
        is_cap = True
    elif is_cap:
        j = 0
        print(bb[i-j][1] - bb[i-j- 1][3])
        while i-j>0 and bb[i-j][1] - bb[i-j- 1][3]> 2:
            if text[j].split(' ')[0] == 'Figure':
                is_cap =  True
            else:
                is_cap = False
            j = j + 1
    return is_cap

lab  = ['title', 'author', 'sec', 'meta', 'para', 'fnote', 'page', 'other']
def main(pdf_path, path ,save_path):
    is_cap = False
    json_file = 'yolo_hrds_4_gt_test/json/ACL_2020.acl-main.99.json'#'dataset_parse/json/2022.naacl-main.92.json'
    name_im = 'ACL_2020.acl-main.99'#.json#'2022.naacl-main.92'
    with open(json_file) as f:
        data = json.load(f)
        bb, page, text, size, type, style, font = get_info_json(data)
        labels = []
        edge = []
        for i in range(len(bb)): # Ogni bb è un mio nodo
            if type[i] == 'text':
                #(page[i] == 0 and bb[i][1]<300 and 'NimbusRomNo9L' not in font[i]) \
                
                if style[i] == 'bold' and size[i] >= 14 and page[i] == 0: # é titolo/ sec .. 
                   # y0 in alto 
                        labels.append(0) #  Title 
                elif (page[i] == 0 and bb[i][1]<200 ) \
                    or (size[i]<10 and bb[i][1]>780):
                        labels.append(2) # meta ? -type diversi
                        print('qui', page[i])
                elif style[i] == 'bold' and 10<= size[i] <= 12 and 'NimbusRomNo9L' in font[i]: # SEC # -> vedi numero prima (?)
                        if is_number(text[i][0]) \
                            or (text[i] in ['Abstract', 'References', 'Acknowledgments']) \
                                or (i>0 and labels[-1]==1 and bb[i][0]-bb[i-1][0]>=10):
                            labels.append(1) #  Sec
                        else: #para
                            labels.append(4) 
                        # elif style[i] == 'italic' and (bb[i][1]<150 or bb[i][3]>700):
                #         labels.append(3) # meta o header
               
                else:
                    is_cap = is_caption(bb, text, i, is_cap)
                    print(is_cap)
                    if 9.5<= size[i] <= 12 and 'NimbusRomNo9L' in font[i] and not is_cap: # Para # -> vedi numero prima (?)
                        labels.append(4) #
                    elif size[i]< 9 and bb[i][1] >700:
                        labels.append(5) # note
                    elif is_number(text[i]) and bb[i][1]> 700: # pagina
                        labels.append(6) # Page
                    elif size[i]<10.3 and is_cap: #other
                      #  is_cap = is_caption(bb, text, i, is_cap)
                       # print(is_cap, 'CAP:', text[i])
                        labels.append(3)
                    else: #other
                        labels.append(8)
            elif type[i] == 'fig':
                #other
                labels.append(9)
            elif type[i] == 'equ':
                #other
                labels.append(10)
            else:
                labels.append(11)
        print(len(bb), len(labels))
        draw_all(path, bb, labels, page, name_im, pdf_path)
            





if __name__ == '__main__':
    dir = 'acl_anthology_pdfs/'
    pdf = '2023.acl-long.150.pdf'#'2022.naacl-main.92.pdf'#
    save_path = 'plot_bb_parse/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = '../dataset_hrds_4class/test/images/'#'dataset_parse/images/'
    pdf_path = dir + pdf
    main(pdf_path, path ,save_path)


    

