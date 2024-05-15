import os
import json
from PIL import Image, ImageDraw

from pdfParser import bb_scale, dimension_pdf


lettere_caps = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
lab  = ['title', 'sec', 'meta', 'caption' , 'para', 'note', 'page','fig', 'equ', 'tab', 'alg', 'other']

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
    elif lab == 8:
        color = 'magenta'
    elif lab == 9:
        color = 'pink'
         #color = '#ECEBBD'
    else:
        color = 'black'

    return color


def draw_all(path, bb, lab, page, name_im, wp, hp):
    #name = path + name_im + '/' + name_im + '_0.png' # iniz.
    name = path + name_im + '_0.png' # iniz.

    image = Image.open(name)
    draw = ImageDraw.Draw(image)
    w , h = image.size
 #   print(w,h)
    
 #   print('qqq' , image.size)
    #700 : 842 = x : 792
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

def is_caption(bb, text, i, is_cap, size):
    lett = text[i].split(' ')
    if (lett[0] == 'Figure' or lett[0] == 'Table')\
          and i >0 and (abs(bb[i][1]-bb[i-1][3])>30)\
            and size[i]<10.5: #l'immagine o tabella non stanno accanto
        is_cap = True
    elif is_cap:
        j = 0
        while i-j>0 and abs(bb[i-j][1] - bb[i-j- 1][3])> 4:
            
            if text[i-j-1].split(' ')[0] == 'Figure':
                is_cap =  True
            else:
                is_cap = False
            j = j + 1
        if abs(bb[i][1] - bb[i-1][3])> 5 or  size[i]>10.4:
            is_cap = False
    return is_cap

def is_fnote(bb, text, i, is_note, size):
    #lett = text[i].split(' ')
    if is_number(text[i][0])\
          and i >0 and size[i]<9.1: #l'immagine o tabella non stanno accanto
        is_note = True
    elif is_note:
        j = 0
        while i-j-1<len(text) and abs(bb[i-j][1] - bb[i-j- 1][3])> 4:
            #print('testo', (text[i-j-1]))
            if len(text[i-j-1])>0 and  is_number(text[i-j-1][0]):
                is_note =  True
            else:
                is_note = False
            j = j + 1
        if abs(bb[i][1] - bb[i-1][3])> 5 or  size[i]>9:
            is_note = False
    return is_note

def main(path ,save_path):
    folder = 'yolo_hrds_4_gt_test/'

    folder_json = folder + 'json/'
    list_json = os.listdir(folder_json)
    for j in list_json:
         #760 : 842 = x : 792
        p1 = '_'.join(j.split('_')[1:])
        pdf = p1.replace('json', 'pdf')
        pdf_path = 'acl_anthology_pdfs_test/' + pdf 
        wp, hp = dimension_pdf(pdf_path)
        lim_note = (760*hp)/842

        is_cap = False
        start_text = False
        print(j)
        #NAACL_2021.naacl-main.381.json' #
        json_file = folder_json + j
        #'yolo_hrds_4_gt_test/json/EMNLP_D19-1317.json'
        #ACL_P19-1303.json'#ACL_2020.acl-main.99.json'#'dataset_parse/json/2022.naacl-main.92.json'
        name_im = j.replace('.json', '')
        #'EMNLP_D19-1317' #'ACL_P19-1303'#'ACL_2020.acl-main.99'#.json#'2022.naacl-main.92'
        data = []
        new_data = []
        with open(json_file) as f:
            data = json.load(f)
            bb, page, text, size, type, style, font = get_info_json(data)
            labels = []
            edge = []
            print(text[0])
            for i in range(len(bb)): # Ogni bb è un mio nodo
                
                if type[i] == 'text':
                    #(page[i] == 0 and bb[i][1]<300 and 'NimbusRomNo9L' not in font[i]) \
                    if text[i] == 'Abstract' and style[i] == 'bold':
                        start_text = True
                    if style[i] == 'bold' and size[i] >= 14 and page[i] == 0: # é titolo/ sec .. 
                    # y0 in alto 
                            labels.append(0) #  Title 
                    elif (page[i] == 0 and bb[i][1]<250 and not start_text) \
                        or (size[i]<10 and bb[i][1]>lim_note):
                            labels.append(2) # meta ? -type diversi
                    elif style[i] == 'bold' and 9.5<= size[i] <= 12 \
                        and ('NimbusRomNo9L' in font[i] or 'Times' in font[i]): # SEC # -> vedi numero prima (?)
                            # if is_number(text[i][0]) \
                            #     or (text[i] in ['Abstract', 'References', 'Acknowledgments','Acknowledgment', 'Appendix']) \
                            #         or (i>0 and labels[-1]==1 and bb[i][0]-bb[i-1][0]>=10) \
                            #     or (len(text[i])>2 and text[i][0] in lettere_caps and text[i][1] in [' ', '.']):
                            if text[i][0] != '•' :      
                                labels.append(1)    #  Sec
                                    #break
                            else:                   #para
                                labels.append(4)
                    else:
                        is_cap = is_caption(bb, text, i, is_cap, size)
                      #  is_note = is_fnote(bb, text, i, is_note, size)
                        if is_number(text[i]) and bb[i][1]> lim_note: # pagina
                            labels.append(6)        # Page
                        elif 9.1<= size[i] <= 12 and \
                            ('NimbusRomNo9L' in font[i] or 'Times' in font[i] or 'NimbusMonL' in font[i]) and not is_cap:# and not is_note: # Para # -> vedi numero prima (?)
                            labels.append(4) #
                        elif lim_note > bb[i][1] >500 and not is_cap and size[i]<9.1 :#and ('NimbusRomNo9L' in font[i] or 'Times' in font[i]):# and is_note:
                            labels.append(5)        # note

                        elif size[i]<10.3 and is_cap: #caption
                            labels.append(3)

                        else: #other
                            labels.append(10)
                elif type[i] == 'fig':
                    #other
                    labels.append(7)
                elif type[i] == 'equ':
                    #other
                    labels.append(8)
                elif type[i] == 'tab':
                    labels.append(9)
                elif type[i] == 'alg':
                    labels.append(9)
                else:
                    labels.append(10)
           # print(len(bb), len(labels))
            draw_all(path, bb, labels, page, name_im, wp, hp)
            print(len(new_data))
            for i in range(len(bb)):
                dict = {'box': bb[i], 
                        'style': style[i], 
                        'size': size[i], 
                        'font': font[i], 
                        'page': page[i], 
                        'text': text[i], 
                        'type': type[i],
                        'class': lab[labels[i]]}
                new_data.append(dict)

            save_json = folder +'check_json_label/' + j 
            if not os.path.exists(folder +'check_json_label/'):
                os.makedirs(folder +'check_json_label/')
            with open(save_json, 'w') as f:
                json.dump(new_data, f, indent=4)

                
if __name__ == '__main__':
   # dir = 'acl_anthology_pdfs/'
   # pdf = '2022.naacl-main.92.pdf'#2023.acl-long.150.pdf'# #solo per ridimensionare imm.
    save_path = 'plot_bb_parse/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = '../dataset_hrds_4class/test/images/'#'dataset_parse/images/'
   # pdf_path = dir + pdf
    main(path ,save_path)


    

