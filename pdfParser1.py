import io
import os
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import fitz
import json
import pdfplumber
from overlap_box_Yolo import check_overlap
def combine_bb_image(bb): 
    comb = False
    new_bb = []

    for i in range(len(bb)):

        if i < len(bb)-1 and (bb[i+1][1]) - 20 <=(bb[i][1]) <= (bb[i+1][1]) + 20 \
            and (((bb[i+1][0]) - 20 <= (bb[i][2]) <= (bb[i+1][0]) + 20) \
                 or ((bb[i+1][0]) - 25 <= (bb[i][0]) <= (bb[i+1][0]) + 25)):
            if comb: # ha fatto gia dei merge
                x0 = min(x0, bb[i+1][0])
                y0 = min(y0, bb[i+1][1])
                x1 = max(x1, bb[i+1][2])
                y1 = max(y1, bb[i+1][3])
            else:                  
                x0 = min(bb[i][0], bb[i+1][0])
                y0 = min(bb[i][1], bb[i+1][1])
                x1 = max(bb[i][2], bb[i+1][2])
                y1 = max(bb[i][3], bb[i+1][3])
                comb = True
        else:
            if comb:
                new_bb.append([x0, y0, x1, y1])
                #f_style, f_size, font, text
                comb = False
            else:
                new_bb.append(bb[i])
    return new_bb   

def combine_bb(bb, f_style, f_size, font, text, data, k): 
    #stessa y0 e vicini x(0)2 == x(1)0  (?) 
    #x0, y0, x1, y1 = bb[0]
    comb = False
    new_bb = []
    
    # image = Image.open(name)
    # draw = ImageDraw.Draw(image)
    # w , h = image.size
    # wp, hp = dimension_pdf(pdf_path)
    new_text = ''
    #dict_data = {'box' : , 'style': , 'size': , 'font': , 'text': }
    for i in range(len(bb)):

        if i < len(bb)-1 and (bb[i+1][1]) - 4 <=(bb[i][1]) <= (bb[i+1][1]) + 4 \
            and (((bb[i+1][0]) - 10 <= (bb[i][2]) <= (bb[i+1][0]) + 10) \
                 or ((bb[i+1][0]) - 10 <= (bb[i][0]) <= (bb[i+1][0]) + 10)):
            if comb: # ha fatto gia dei merge
                x0 = min(x0, bb[i+1][0])
                y0 = min(y0, bb[i+1][1])
                x1 = max(x1, bb[i+1][2])
                y1 = max(y1, bb[i+1][3])
                new_text = new_text + ' ' + text[i+1]
            else:   
                new_text = text[i] + ' ' + text[i+1]
                
                x0 = min(bb[i][0], bb[i+1][0])
                y0 = min(bb[i][1], bb[i+1][1])
                x1 = max(bb[i][2], bb[i+1][2])
                y1 = max(bb[i][3], bb[i+1][3])
                comb = True
            #print(new_text)
          #  draw.rectangle(bb_scale([x0, y0, x1, y1], w, h, float(wp), float(hp)), outline = 'cyan') 
           
        else:
            if comb:
                new_bb.append([x0, y0, x1, y1])
                #f_style, f_size, font, text
                dict_data = get_dict([x0, y0, x1, y1], f_style[i], f_size[i], font[i], k, new_text, 'text')
                #{'box' : [x0, y0, x1, y1], 'style': f_style[i], 'size': f_size[i], 'font': font[i], 'text': new_text, 'page': k, 'type': 'text'}
                comb = False
            else:
                dict_data = get_dict(bb[i], f_style[i], f_size[i], font[i], k, text[i], 'text')
                #{'box' : bb[i], 'style': f_style[i], 'size': f_size[i], 'font': font[i], 'text': text[i], 'page': k, 'type': 'text'}
                new_bb.append(bb[i])
            #if text[i] != ' ' or new_text != ' ':
            data.append(dict_data)
            new_text = ''    
    return data    

def parse_pymupdf(path, name, save):
    doc = fitz.open(path)
    data = [] # metto tutte le pagine nello stesso json

    for k in range(len(doc)):
        page = doc[k] # Access page
        all_infos = page.get_text("dict", sort=True)#, flags=11)
    
        bb = []
        f_size = []
        f_style = []
        font = []
        text = []           

        #{lines,lines_strict,text,explicit}horizontal_strategy="lines_strict",
        tables = page.find_tables(intersection_tolerance=45)#, join_tolerance =1)#, snap_x_tolerance= 15)#horizontal_strategy='text', vertical_strategy='text')
        table_data = []
        for t in tables:
            dict_tab = get_dict(t.bbox, False, False, False, k, t.extract(), 'tab')
            table_data.append(dict_tab)

        d = page.get_text("dict")
        blocks = d["blocks"]  # the list of block dictionaries
        imgblocks = [b for b in blocks if b["type"] == 1]
        image_data = []
        for im in imgblocks: # TODO MERGE
            #box_im = 
            dict_im = get_dict(im['bbox'], False, False, False, k, False, 'img')
            image_data.append(dict_im)
            #print(dict_im)
       
        tmp = []
        for block in all_infos['blocks']:
            if 'lines' in block:
                for line in block['lines']:
                    for span in line['spans']:
                        flags = span['flags']
                        #print(span['flags'], '\n')
                        style = get_style(flags)
                        
                        bb.append(span['bbox'])   
                        f_size.append(span['size'])
                        f_style.append(style)
                        font.append(span['font'])
                        text.append(span['text'])

        tmp = combine_bb(bb, f_style, f_size, font, text,  tmp, k)
        print('\n ###PAGE: ', k, len(image_data))
        tmp = get_caption(tmp, image_data, table_data)
        data = data + tmp # Metto immagini in fondo alla pagina ! ! 
    
    save_json = save +'json/' + name +'.json' 
    with open(save_json, 'w') as f:
        json.dump(data, f, indent=4)

    return data

def find_images_bbox(doc, page, k):
    image_list = doc.get_page_images(k, full=True)
    for i in range(len(image_list)):
        image_bbox = page.get_image_bbox(image_list[i])
        print('image {} Bbox: {}'.format(i, image_bbox))

def get_caption(data, image_data, table_data):
    cap_im_tab = []
    new_data = []
    count_im = 0
    count_tab = 0
    for i in data: # in questo modo dopo immagine o tabella ho la caption associata
        #print(i)
        # Y1 dell'imm simile a Y0 del tab && X0 simile a X0
        #if count_tab< len(table_data) :
        #  print(check_overlap(i['box'], table_data[count_tab]['box']), i['text'])
        if 'Figure' in i['text'] and i['size']<10.2 and \
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
            if (count_tab< len(table_data) \
                 and check_overlap(i['box'], table_data[count_tab]['box'])==0.0):
                 new_data.append(i) # Tolgo da dentro i caption di fig e tab
            elif count_tab>= len(table_data):# or (len(table_data)== 1 and count_im==0 ): # non ci sono piu tabelle
                new_data.append(i)

    while count_im < len(image_data):
        cap_im_tab.append(image_data[count_im])
        count_im = count_im + 1

   # print(len(image_data))
    new_data = new_data + cap_im_tab
    return new_data

def get_dict(box, style, size, font, page, text, type):
    return {'box': box, 'style': style, 'size': size, 'font': font, 'page': page, 'text': text, 'type': type}

def get_style(flags):
    if  bool(flags & 2**4):
        style = 'bold'
    elif bool(flags & 2**1):
        style = 'italic'
    else:
        style = 'normal'
    return style

def get_image(file, save, name):
    # Convert PDF to images
    images = convert_from_path(file)

    # Save each page as an image

    if not os.path.exists(save +'images/' +name):
        os.makedirs(save + 'images/' + name)

    for i in range(len(images)):
        images[i].save(save +'images/' +name + '/' +name + '_' +str(i) + '.jpg', 'JPEG')  
     
def draw_bb_one(pdf, bounding_box, image_bboxes):
    name = 'page0.jpg'
    image = Image.open(name)
    draw = ImageDraw.Draw(image)
    w , h = image.size
    wp, hp = dimension_pdf(pdf_path)
    print('qqq' , image.size)
    i = 0

    for bb in bounding_box:
        draw.rectangle(bb_scale(bb, w, h, float(wp), float(hp)), outline = 'black') 
        #image.save('zzz/page0_bb'+ str(i) +'.jpg')
        i = i + 1
    for im in image_bboxes:
        draw.rectangle(bb_scale(im, w, h, float(wp), float(hp)), outline = 'purple' , width=3) 

    image.save('page0_bb.jpg')

def get_color(type):
    if type == 'img':
        color = 'magenta'
    elif type == 'tab':
        color = 'purple'
    else:
        color = 'black'
    return color 

def draw_all(path, data, name_im):
    name = path + name_im + '_0.jpg' # iniz.

    image = Image.open(name)
    draw = ImageDraw.Draw(image)
    w , h = image.size
    wp, hp = dimension_pdf(pdf_path)
    print('qqq' , image.size)
    i = 0
    for d in data:
        
        
        if i >0 and data[i]['page'] != data[i-1]['page'] :
            #path_save = 'bb_check/' + name_im + '_' +str(data[i-1]['page']) + '.jpg'
            path_save = path + name_im + '_' +str(data[i-1]['page']) + '.jpg'

            image.save(path_save)

            name = path + name_im + '_' + str(data[i]['page']) + '.jpg'
            image = Image.open(name)
            draw = ImageDraw.Draw(image)
        color = get_color(data[i]['type'])
        draw.rectangle(bb_scale(d['box'], w, h, float(wp), float(hp)), outline = color) 
        #image.save('zzz/page0_bb'+ str(i) +'.jpg')
        
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
        # Apri il file PDF in lettura binaria
    with open(pdf_path, 'rb') as file:
        # Crea un oggetto PdfFileReader
        reader = PdfReader(file)
        
        # Ottieni la prima pagina (indice 0)
        page = reader.pages[0]
        
        # Ottieni le dimensioni della pagina
      #  print(page.mediabox)
        width = page.mediabox[2]
        height = page.mediabox[3]
        
        # Stampa le dimensioni
      #  print(f"Width = {width}, Height = {height}")
    return width, height

if __name__ == '__main__':
    # Example usage
    # Directory to save the PDFs
    dir = 'acl_anthology_pdfs/'
    save_path = 'dataset_parse/'

    list_doc = os.listdir(dir) # Ciclare su PDF TODO
   # pdf = list_doc[2]
    #pdf ='2022.naacl-demo.0.pdf'
    for pdf in list_doc:
       # pdf = '2023.acl-long.150.pdf'#'2022.naacl-main.92.pdf'#
        print(pdf)
        pdf_path = dir + pdf

        name = pdf.replace('.pdf', '')
        
        # Get image from PDF
        get_image(pdf_path, save_path, name)
        data = parse_pymupdf(pdf_path, name, save_path)

        draw_all(save_path + 'images/' + name +'/', data, name)
        #bb, f_style, f_size, font, text, image_bboxes = parse_pymupdf(pdf_path)
        #draw_bb(pdf,bb, image_bboxes)
        #data = combine_bb(bb, f_style, f_size, font, text)
        
        # save_json = 'json_parse/' + 'prova.json' 
        # with open(save_json, 'w') as f:
        #     json.dump(data, f, indent=4)
        #draw_bb(pdf, new_bb)


# Estrarre immagini -> folder / image / image0.png
# Creare Json

# |json  -> / name.json
# |image -> / name / name_0.png