import io
import os
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import fitz
import json

def combine_bb(bb, f_style, f_size, font, text, data, k): 
    #stessa y0 e vicini x(0)2 == x(1)0  (?) 
    x0, y0, x1, y1 = bb[0]
    comb = False
    new_bb = []
    
    # image = Image.open(name)
    # draw = ImageDraw.Draw(image)
    # w , h = image.size
    # wp, hp = dimension_pdf(pdf_path)
    new_text = ''
    #dict_data = {'box' : , 'style': , 'size': , 'font': , 'text': }
    for i in range(len(bb)):

        if i < len(bb)-1 and (bb[i+1][1]) - 2 <=(bb[i][1]) <= (bb[i+1][1]) + 2 \
            and (((bb[i+1][0]) - 2 <= (bb[i][2]) <= (bb[i+1][0]) + 2) \
                 or ((bb[i+1][0]) - 2 <= (bb[i][0]) <= (bb[i+1][0]) + 2)):
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
                dict_data = {'box' : [x0, y0, x1, y1], 'style': f_style[i], 'size': f_size[i], 'font': font[i], 'text': new_text, 'page': k}
                comb = False
            else:
                dict_data = {'box' : bb[i], 'style': f_style[i], 'size': f_size[i], 'font': font[i], 'text': text[i], 'page': k}
                new_bb.append(bb[i])
            data.append(dict_data)
            new_text = ''    
    return data

#(x0, y0) is the bottom-left corner of the bounding box, and (x1, y1) is the top-right corner.
def parse_pymupdf(path, name, save):
    doc = fitz.open(path)
    data = [] # metto tutte le pagine nello stesso json
    
    # TODO itero su Pagine ..
    for k in range(len(doc)):
        page = doc[k]#[0] # Access the first page (0-based index)
        all_infos = page.get_text("dict", sort=True)#, flags=11)
        #text = page.get_text() # Extract plain text
    #  print(text) # Print the extracted text
        bb = []
        f_size = []
        f_style = []
        font = []
        text = []
        image_bboxes = []
        
        d = page.get_text("dict")
        blocks = d["blocks"]  # the list of block dictionaries
        imgblocks = [b for b in blocks if b["type"] == 1]
        #print(imgblocks[0]['bbox'])
        #image_bboxes.append(imgblocks[0]['bbox'])
        image_data = []
        for im in imgblocks:
            dict_im = {'box': im['bbox'], 'height': im['height'], 'width': im['width']}
            print(dict_im)
            image_data.append(dict_im)

        for block in all_infos['blocks']:
            if 'lines' in block:
                for line in block['lines']:
                    for span in line['spans']:
                        flags = span['flags']
                        #print(span['flags'], '\n')
                        style = get_style(flags)
                        #is_bold = bool(flags & 2**4) # Check if the font is bold
                        #is_italic = bool(flags & 2**1) # Check if the font is italic  

                        bb.append(span['bbox'])   
                        f_size.append(span['size'])
                        f_style.append(style)
                        font.append(span['font'])
                        text.append(span['text'])
                    # print(f"Font: {span['font']}, Size: {span['size']}, Bounding Box: {span['bbox']}, Text: {span['text']}, Style: {style}")
        data = combine_bb(bb, f_style, f_size, font, text, data, k)

        
    save_json = save +'json/' + name +'.json' 
    with open(save_json, 'w') as f:
        json.dump(data, f, indent=4)

    return data #bb, f_style, f_size, font, text, image_bboxes, image_data

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
     
def draw_bb(pdf, bounding_box, image_bboxes):
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
        draw.rectangle(bb_scale(im, w, h, float(wp), float(hp)), outline = 'purple') 

    image.save('page0_bb.jpg')

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
        print(page.mediabox)
        width = page.mediabox[2]
        height = page.mediabox[3]
        
        # Stampa le dimensioni
        print(f"Width = {width}, Height = {height}")
    return width, height

if __name__ == '__main__':
    # Example usage
    # Directory to save the PDFs
    dir = 'acl_anthology_pdfs/'
    save_path = 'dataset_parse/'
   # list_doc = os.listdir(save_dir) # Ciclare su PDF TODO
   # pdf = list_doc[2]
    pdf ='2022.naacl-demo.0.pdf'

    print(pdf)
    pdf_path = dir + pdf

    name = pdf.replace('.pdf', '')
    
    # Get image from PDF
    get_image(pdf_path, save_path, name)
    parse_pymupdf(pdf_path, name, save_path)
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