import os
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import fitz
import json
from overlap_box_Yolo import check_overlap
import re
from collections import Counter

def split_string(stringa):
    # Sostituisci ogni carattere maiuscolo con uno spazio
    stringa_con_spazi = ''.join(' ' + c if c.isupper() else c for c in stringa)
    
    # Dividi la stringa in base allo spazio
    parole = stringa_con_spazi.split()
    
    # Rimuovi gli spazi aggiuntivi
    parole = [parola.strip() for parola in parole]
    
    return parole

def get_font_comm(font):
    fc1 = (Counter(font).most_common()[0][0].split('-')[0])
    fc2 = split_string(fc1)
    if len(fc2[0]) == 1:
        font_comm = fc2[0] + fc2[1]
    else:
        font_comm = fc2[0]
    return font_comm

# Deprecated (?) -> YOLO
def get_caption_tab(data, image_data, table_data):
    cap_im_tab = []
    new_data = []
    count_im = 0
    count_tab = 0
    for i in data: # in questo modo dopo immagine o tabella ho la caption associata
        #print(i)
        # Y1 dell'imm simile a Y0 del tab && X0 simile a X0
        #if count_tab< len(table_data) :
        #  print(check_overlap(i['box'], table_data[count_tab]['box']), i['text'])
        if 'Figure' in i['text'] and i['size'] < 10.2 and \
            count_im < len(image_data) and \
                image_data[count_im]['box'][3]- 20 <  i['box'][1] < image_data[count_im]['box'][3]+ 20:
             #       and image_data[count_im]['box'][0]-10 <  i['box'][0] < image_data[count_im]['box'][0]+10: 
                cap_im_tab.append(image_data[count_im])
                cap_im_tab.append(i)
                
                print('|Fig -> ', i['text'] , ':', i['box'], '|', image_data[count_im]['box'])
                count_im = count_im + 1
        elif count_tab < len(table_data) and 'Table' in i['text'] and i['size']<10.2 and i['box'][1]>= table_data[count_tab]['box'][1] -15:#and i['page'] == table_data[count_tab]['page'] :#and \
               # count_tab < len(table_data) and \
               # table_data[count_tab]['box'][3]-50 <  i['box'][1] < table_data[count_tab]['box'][3]+50:
                cap_im_tab.append(table_data[count_tab])
                cap_im_tab.append(i)
                #print('|Tab -> ', i['text'],  table_data[count_tab]['box'])
                count_tab = count_tab + 1
        else:
            if count_tab< len(table_data) \
                 and check_overlap(i['box'], table_data[count_tab]['box'])==0.0:
                 new_data.append(i) # Tolgo da dentro i caption di fig e tab
            elif count_tab>= len(table_data):# or (len(table_data)== 1 and count_im==0 ): # non ci sono piu tabelle
                new_data.append(i)

    while count_im < len(image_data):
        cap_im_tab.append(image_data[count_im])
        count_im = count_im + 1

    new_data = new_data + cap_im_tab
    return new_data

def check_style(bb, style, ind1, ind2, text, font, size):
    
    if style[ind1] in ['normal','italic'] and style[ind2] in ['normal','italic']:
        return True
    elif style[ind1] == style[ind2]:
         return True
    elif style[ind1] != 'bold' and style[ind2] == 'bold':# and get_width(bb[ind1])<get_width(bb[ind2]):
        #print(text[ind1], '|', text[ind2], 'Y', bb[ind2][0]-bb[ind1][2])
        return True
    elif style[ind2] != 'bold' and style[ind1] == 'bold' \
        and len(text[ind2].replace(' ', ''))>0 \
        and not text[ind2].replace(' ', '')[0].isupper() and not text[ind2].replace(' ', '')[0] == '.':# and get_width(bb[ind2])<get_width(bb[ind1]):# and :
        # if text[ind2].replace(' ', '')[0].isupper():
        
        return True
        
    return False

def get_width(box):
    return abs(box[2]- box[0])

def get_higth(box):
    return abs(box[3]-box[1])

def get_font_word(text, font):
    new_font = []
    for i in range(len(text)):
        t1 = text[i].split(' ')
        for k in t1:
            new_font.append(font[i])
    return new_font

def combine_bb(bb, f_style, f_size, font, text, block, k): 
    comb = False
    new_bb = []
    data = []

    font_split = get_font_word(text, font)
    comm_f = get_font_comm(font_split)
    # image = Image.open(name)
    # draw = ImageDraw.Draw(image)
    # w , h = image.size
    # wp, hp = dimension_pdf(pdf_path)
    new_text = ''
    #dict_data = {'box' : , 'style': , 'size': , 'font': , 'text': }
    for i in range(len(bb)):
       # y simile
       # x simil            
            # and '•' not in new_text
        if i < len(bb)-1 and \
            ((check_overlap(bb[i+1], bb[i])>0.005 and \
              (bb[i][2]-bb[i][0]<10 or bb[i+1][2]-bb[i+1][0]<10 \
               or (abs(bb[i+1][1]-bb[i][1])<3  and abs(bb[i+1][3]-bb[i][3])<3))) \
                or((abs(bb[i+1][1] - bb[i][1]) <= 5 or abs(bb[i+1][3] - bb[i][3]) <=10)
            and ((abs(bb[i+1][0] - bb[i][2]) <= 20 and check_style(bb, f_style, i, i+1, text, font, f_size)))\
                or ((abs(bb[i+1][0] - bb[i][2])<=4 and (bb[i][2]-bb[i][0]<10 or bb[i+1][2]-bb[i+1][0]<10 ))))):#\
                 #   or (abs(bb[i+1][0] - bb[i][2]) <= 5 and f_style[i] == 'bold')):# \
                   #(abs(bb[i+1][0] - bb[i][2]) <= 10) or\
           
            if comb: # ha fatto gia dei merge
                
                if bb[i+1][2]-bb[i+1][0]<11 and contiene_simboli_speciali(text[i+1]):
                    y0 = y0
                    y1 = y1
                elif x1-x0<11 and contiene_simboli_speciali(new_text):
                    y0 = (bb[i+1][1])
                    y1 = (bb[i+1][3])
                else:
                    y1 = max(y1, bb[i+1][3])
                    y0 = min(y0, bb[i+1][1])
                x0 = min(x0, bb[i+1][0])                
                x1 = max(x1, bb[i+1][2])
                
                new_text = new_text + ' ' + text[i+1]
            else:   
                new_text = text[i] + ' ' + text[i+1]

                if bb[i+1][2]-bb[i+1][0]<11 and contiene_simboli_speciali(text[i+1]):
                    y0 = (bb[i][1])
                    y1 = (bb[i][3])
                elif bb[i][2]-bb[i][0]<11 and contiene_simboli_speciali(text[i]):
                    y0 = (bb[i+1][1])
                    y1 = (bb[i+1][3])
                else:
                    y0 = min(bb[i][1], bb[i+1][1])
                    y1 = max(bb[i][3], bb[i+1][3])
                x0 = min(bb[i][0], bb[i+1][0])
                x1 = max(bb[i][2], bb[i+1][2])
                comb = True

            
            #or ('CMMI10' in font[i]  and comm_f not in font[i+1]))
            if (f_style[i] in ['normal','italic'] \
                and (comm_f in font[i] )\
                    and f_size[i]> f_size[i+1]-1.5) \
                        or (f_style[i] == f_style[i+1] and f_size[i+1] < f_size[i]-0.5) \
                        or (f_style[i] != f_style[i+1] and bb[i+1][2]-bb[i+1][0]<5):
                f_style[i+1] = f_style[i]
                f_size[i+1] = f_size[i]
                font[i+1] = font[i]   
            bb[i+1] = [x0, y0, x1, y1]
            text[i+1] = new_text
          
        else:
            if comb:
                new_bb.append([x0, y0, x1, y1])
                dict_data = get_dict(block[i], [x0, y0, x1, y1], f_style[i], f_size[i], font[i], k, new_text, 'text')
                comb = False                        
            else:
                dict_data = get_dict(block[i], bb[i], f_style[i], f_size[i], font[i], k, text[i], 'text')
                new_bb.append(bb[i])
           # print(dict_data)
            data.append(dict_data)
            new_text = ''    
    return data 
   
def contiene_simboli_speciali(stringa):
    # Pattern per cercare qualsiasi carattere che non sia una lettera, un numero o uno spazio
    pattern = re.compile('[^a-zA-Z0-9 ]')
    return bool(pattern.search(stringa))

def get_text(all_infos, bb, f_size, f_style, font, text, block_array, k):
    tmp = []
    conunt_block = 0
     
    for block in all_infos['blocks']:
        if 'lines' in block:
            for line in block['lines']:
                c = 0
                for span in line['spans']:
                    flags = span['flags']
                        #print(span['flags'], '\n')
                    style = get_style(flags)

                    if (span['size']< (span['bbox'][3]-span['bbox'][1]) \
                        and (span['bbox'][3]-span['bbox'][1]-span['size'])>6):
                        p_m = (span['bbox'][3]+span['bbox'][1])/2
                        if c > 0 and (abs(line['spans'][c-1]['bbox'][2] - span['bbox'][0]))<10 :
                            y0 = line['spans'][c-1]['bbox'][1] - 2 # se quello precedente è attaccato
                            y1 = line['spans'][c-1]['bbox'][3] 
                        else:
                            y0 = p_m - 2
                            y1  = p_m + 1
                        span['bbox'] = [span['bbox'][0],y0,span['bbox'][2],y1]
                    bb.append(span['bbox'])#span['bbox'])   
                    f_size.append(span['size'])
                    f_style.append(style)
                    font.append(span['font'])
                    text.append(span['text'])
                    block_array.append(block['number'])

                    dict_data = get_dict(block['number'], span['bbox'], style, span['size'], span['font'], k, span['text'], 'text')
                    tmp.append(dict_data)
                    c= c+1
        conunt_block = conunt_block + 1
    return tmp, bb, f_style, f_size, font, text, block_array

def get_line(k, page):
    draw = []
    paths = page.get_drawings()
    for path in paths:
            # print(path)
        rect = path["rect"]
        bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
        if rect.y0 == rect.y1:
            dict_im = get_dict(bbox, False, False, False, k, False, 'linee')
        else:
            dict_im = get_dict(bbox, False, False, False, k, False, 'draw')
        draw.append(dict_im)
        print(f"k:: {k}, Bounding box: {bbox}")
    return draw

def get_images(k, page):
    d = page.get_text("dict")
    blocks = d["blocks"]  # the list of block dictionaries
    imgblocks = [b for b in blocks if b["type"] == 1]
    image_data = []
    for im in imgblocks: # TODO MERGE
            #box_im = 
        dict_im = get_dict([], im['bbox'], False, False, False, k, False, 'fig')
        image_data.append(dict_im)
    return image_data

def get_tables(k, page):
    #{lines,lines_strict,text,explicit}horizontal_strategy="lines_strict",
    tables = page.find_tables(intersection_tolerance=40)#, join_tolerance =1)#, snap_x_tolerance= 15)#horizontal_strategy='text', vertical_strategy='text')
    table_data = []
    for t in tables:
        dict_tab = get_dict([], t.bbox, False, False, False, k, t.extract(), 'tab')
        table_data.append(dict_tab)
    return table_data

# altro modo per trovare immagini
def find_images_bbox(doc, page, k):
    image_list = doc.get_page_images(k, full=True)
    for i in range(len(image_list)):
        image_bbox = page.get_image_bbox(image_list[i])
        print('image {} Bbox: {}'.format(i, image_bbox))

def get_dict(block, box, style, size, font, page, text, type):
    return {'block': block, 'box': box, 'style': style, 'size': size, 'font': font, 'page': page, 'text': text, 'type': type}

def get_style(flags):
    if  bool(flags & 2**4):
        style = 'bold'
    elif bool(flags & 2**1):
        style = 'italic'
    else:
        style = 'normal'
    return style

# Convert PDF to images

def get_image(file, save, name):
    images = convert_from_path(file)
    width = 596
    height = 842

    check_folder(save +'images_TESI/' +name)
    print(save +'images_TESI/' +name)
    for i in range(len(images)):
       # images[i].save(save +'images/' +name + '/' +name + '_' +str(i) + '.jpg', 'JPEG') 
        img_resized = images[i].resize((width, height), Image.LANCZOS)
       # images[i].save(save +'image_test/' +name + '_' +str(i) + '.jpg', 'JPEG') 
        print(save +'images_TESI/' +name + '_' +str(i) + '.jpg')
        img_resized.save(save +'images_TESI/' + name +'/' +name + '_' +str(i) + '.jpg', 'JPEG') 

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path) 
  
def get_color(type):
    if type == 'fig':
        color = 'magenta'
    elif type == 'tab':
        color = 'yellow'
    elif type == 'equ':
        color = 'cyan'
    elif type == 'alg':
        color = 'green'
    else:
        color = 'black'
    return color 

def draw_all(path, data, name_im, new_path):
    # inizializzo 
    name = path + name_im + '_0.jpg'
    image = Image.open(name)
    draw = ImageDraw.Draw(image)
    w , h = image.size
    wp, hp = dimension_pdf(pdf_path)

    i = 0
    for d in data:
        print(d['box'])
        if i >0 and data[i]['page'] != data[i-1]['page'] :
            #path_save = 'bb_check/' + name_im + '_' +str(data[i-1]['page']) + '.jpg'
            path_save = 'TESI_mergePY/' + name_im + '_' +str(data[i-1]['page']) + '.jpg'
            #path_save = new_path + name_im + '_' +str(data[i-1]['page']) + '.jpg'

            print('qq',path_save)
            image.save(path_save)

            name = path + name_im + '_' + str(data[i]['page']) + '.jpg'
            image = Image.open(name)
            draw = ImageDraw.Draw(image)

        color = get_color(data[i]['type'])
      #  bb_scale(d['box'], w, h, float(wp), float(hp))
        draw.rectangle(d['box'], outline = color, width=1) 

        i = i + 1

# x: w = xo : 600
 # x: w = xo : 800
def bb_scale(bb, w, h, wp, hp):
    x0 = (bb[0]*w) / wp
    y0 = (bb[1]*h) / hp
    x1 = (bb[2]*w) / wp
    y1 = (bb[3]*h) / hp

    return x0,y0,x1,y1

def dimension_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:

        reader = PdfReader(file)
        page = reader.pages[0]

        width = page.mediabox[2]
        height = page.mediabox[3]
    return width, height

def parse_pymupdf(path, name, save):
    doc = fitz.open(path)
    data = [] # metto tutte le pagine nello stesso json

    for k in range(len(doc)):
        page = doc[k]#[0] # Access pages
        all_infos = page.get_text("dict")#, sort=True)#, flags=11)
    
        bb = []
        f_size = []
        f_style = []
        font = []
        text = [] 
        block = []          

        #table_data = get_tables(k, page)
        #image_data = get_images(k, page)
            #print(dict_im)
        #draw = get_line(k, page)

        tmp_data, bb, f_style, f_size, font, text, block = get_text(all_infos, bb, f_size, f_style, font, text, block, k)
        
        tmp = combine_bb(bb, f_style, f_size, font, text, block, k) 
        imm = get_images(k, page) 
        tab = get_tables(k, page)  
       # #tmp = get_caption_tab(tmp, image_data, table_data) # TODO image and table # Metto immagini in fondo alla pagina ! !
        data = data + tmp#_data  #   + imm + tab #_+ imm
    if not os.path.exists(save +'json_parse/'):
        os.makedirs(save +'json_parse/')
    save_json = save +'json_parse/' + name +'.json' 
    with open(save_json, 'w') as f:
       json.dump(data, f, indent=4)

    return data

if __name__ == '__main__':
    
        dir =  'acl_anthology_pdfs_train/'#'acl_anthology_pdfs/'#_test/'#'acl_anthology_pdfs/'
        save_path = 'TESI_mergePY/'# "Check_cm_Parse/"#'yolo_hrds_4_gt_test/'#'yolo_hrdhs_672_3_testGT2/'#
        check_folder(save_path)
        list_doc = os.listdir(dir) # Ciclare su PDF TODO
        #pdf = list_doc[2]
        #pdf ='2022.naacl-demo.0.pdf'
        print(len(list_doc))

    #for pdf in list_doc:
      #  pdf = '2020.acl-main.99.pdf'#2021.naacl-main.67.pdf'#'P11-1008.pdf' #D13-1134.pdf'
      ##  pdf = '2023.acl-long.150.pdf'#'2022.naacl-main.92.pdf'#
    #    pdf = "2020.acl-main.99.pdf"#"2023.acl-long.150.pdf"#'2022.naacl-demo.4.pdf'
        pdf = '2020.acl-main.233.pdf'#'2020.acl-main.86.pdf'#'2022.naacl-demo.4.pdf'
        print(pdf)
        pdf_path = dir + pdf

        name = pdf.replace('.pdf', '')
        
        # Get image from PDF
        get_image(pdf_path, save_path, name)
        data = parse_pymupdf(pdf_path, name, save_path)
        new_path = 'bb_draw_parse_PERTESI/'
        check_folder(new_path)
        draw_all(save_path + 'images_TESI/' + name +'/', data, name, new_path)

#1355