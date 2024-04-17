import io
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
import pdfminer
import os
from pdfminer.high_level import extract_pages
from PIL import Image, ImageDraw
from pdf2image import convert_from_path


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams(all_texts=True, detect_vertical=True)
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()
    print(text)
    fp.close()
    device.close()
    retstr.close()
   # return text

from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar, LTTextBoxHorizontal, LTTextLineHorizontal
from pdfminer.pdfpage import PDFPage

def extract_text_info(path):
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    fp = open(path, 'rb')

    text_info = []
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    retstr = StringIO()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)
        layout = device.get_result()
       # print(page)
        for element in layout:
            #print(element)
            if isinstance(element, LTTextBox) or isinstance(element, LTTextLine) or isinstance(element, LTTextBoxHorizontal):
                 style_info = {
                    'text': element.get_text(),
                    'position': (element.bbox[0], element.bbox[1])
                }
                 print(style_info)
                # for child in element:
                #     print(child)
                #     if isinstance(child, LTTextLineHorizontal):

                        # Estrai le informazioni di stile e posizione
                        # style_info = {
                        #     'text': child.get_text(),
                        #     'fontname': child.fontname,
                        #     'size': child.size,
                        #     'bold': child.fontname.startswith('Bold'),
                        #     'italic': child.fontname.startswith('Italic'),
                        #     'position': (child.bbox[0], child.bbox[1])
                        # }
                        # text_info.append(style_info)
    text = retstr.getvalue()
    print(text)
    fp.close()
    return text_info

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator, TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def extract_text_and_font_info(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams(all_texts=True)#, detect_vertical=True)
    
    # For text extraction
    #device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    
    # For layout analysis
    layout_device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    
    fp = open(path, 'rb')
    #interpreter = PDFPageInterpreter(rsrcmgr, device)
    layout_interpreter = PDFPageInterpreter(rsrcmgr, layout_device)
    
    for page in PDFPage.get_pages(fp):
        # Process page with TextConverter for text extraction
      #  interpreter.process_page(page)
        
        # Process page with PDFPageAggregator for layout analysis
        layout_interpreter.process_page(page)
        layout = layout_device.get_result()
        
        # Example of accessing font information from layout
        for lobj in layout:
            print(lobj)
            if isinstance(lobj, LTTextBoxHorizontal):
                for text_line in lobj:
                    # Access font information here
                    print('|?' , text_line.get_text())
    
    #text = retstr.getvalue()
    fp.close()
    #device.close()
    retstr.close()
    
   # return text
import fitz


def combine_bb(bb): 
    #stessa y0 e vicini x(0)2 == x(1)0  (?) 
    x0, y0, x1, y1 = bb[0]
    comb = False
    new_bb = []
    name = 'page0.jpg'
    image = Image.open(name)
    draw = ImageDraw.Draw(image)
    w , h = image.size
    wp, hp = dimension_pdf(pdf_path)
    for i in range(len(bb)):
        draw.rectangle(bb_scale(bb[i], w, h, float(wp), float(hp)), outline = 'black') 

        if i < len(bb)-1 and (bb[i+1][1]) - 2 <=(bb[i][1]) <= (bb[i+1][1]) + 2 \
            and (((bb[i+1][0]) - 2 <= (bb[i][2]) <= (bb[i+1][0]) + 2) \
                 or ((bb[i+1][0]) - 2 <= (bb[i][0]) <= (bb[i+1][0]) + 2)):
            if comb:
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
          #  draw.rectangle(bb_scale([x0, y0, x1, y1], w, h, float(wp), float(hp)), outline = 'cyan') 
           
        else:
            if comb:
                new_bb.append([x0, y0, x1, y1])
                comb = False
                draw.rectangle(bb_scale([x0, y0, x1, y1], w, h, float(wp), float(hp)), outline = 'magenta') 
            else:
                new_bb.append(bb[i])
                draw.rectangle(bb_scale(bb[i], w, h, float(wp), float(hp)), outline = 'green') 
       # image.save('zz/prova_'+ str(i) +'.png')
    
    return new_bb



def parse_pymupdf(path):
    doc = fitz.open(path)
    page = doc[0] # Access the first page (0-based index)

    all_infos = page.get_text("dict", sort=True)#, flags=11)
    #text = page.get_text() # Extract plain text
  #  print(text) # Print the extracted text
    bb = []
    for block in all_infos['blocks']:
        if 'lines' in block:
            for line in block['lines']:
                for span in line['spans']:
                    flags = span['flags']
                    print(span['flags'], '\n')
                    is_bold = bool(flags & 2**4) # Check if the font is bold
                    is_italic = bool(flags & 2**1) # Check if the font is italic  
                    bb.append(span['bbox'])        
                    print(f"Font: {span['font']}, Size: {span['size']}, Bounding Box: {span['bbox']}, Text: {span['text']}, Bold: {is_bold}, Italic: {is_italic}")
    return bb

def get_image(file):
    
    # Convert PDF to images
    images = convert_from_path(file)

    # Save each page as an image
    for i in range(len(images)):
        images[i].save('page' + str(i) + '.jpg', 'JPEG')  
     
def draw_bb(pdf, bounding_box):
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
    image.save('page0_bb.jpg')

# x: w = xo : 600
 # x: w = xo : 800

def bb_scale(bb, w, h, wp, hp):
    x0 = (bb[0]*w) / wp
    y0 = (bb[1]*h) / hp
    x1 = (bb[2]*w) / wp
    y1 = (bb[3]*h) / hp

    return x0,y0,x1,y1

from PyPDF2 import PdfReader

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
    save_dir = 'acl_anthology_pdfs/'

   # list_doc = os.listdir(save_dir)
   # pdf = list_doc[2]
    pdf ='2022.naacl-main.3-1.pdf' #'2022.naacl-industry.2.pdf'
    print(pdf)
    pdf_path = save_dir + pdf
   # get_image(pdf_path)
    
    bb = parse_pymupdf(pdf_path)
    draw_bb(pdf,bb)
    new_bb = combine_bb(bb)
    draw_bb(pdf, new_bb)