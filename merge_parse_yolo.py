import os 
import json
from PIL import Image, ImageDraw

from pdfParser import bb_scale, dimension_pdf, get_color, get_dict
from overlap_box_Yolo import check_overlap

""" {
        "box": [
            233.99964999999997,
            661.0002305,
            765.9988259999999,
            1259.0006655
        ],
        "class": "tab",
        "page": "0"
    }
    ####
    {
        "box": [
            201.071044921875,
            74.95138549804688,
            394.5252380371094,
            93.6167221069336
        ],
        "style": "bold",
        "size": 14.346916198730469,
        "font": "NimbusRomNo9L-Medi",
        "page": 0,
        "text": "textless-lib : a Library for",
        "type": "text"
    }
"""

final_bbs = []

def check_box_inside(box_ex, box2_in):
      return box_ex[0] <= box2_in[0] and box_ex[1] <= box2_in[1] \
        and box_ex[2] >= box2_in[2] and box_ex[3] >= box2_in[1]

# yolo_hrdhs_3/jons_yolo/
# dataset_parse/ json
def main():
    json_name = '2020.acl-main.99.json'#'2022.naacl-industry.16.json'
    json_text = 'yolo_hrdhs_672_3_testGT2/json_parse/' + json_name
    json_imm = 'yolo_hrdhs_672_3_testGT2/json_yolo/ACL_2020.acl-main.99.json'# + json_name
    #

    with open(json_text, errors="ignore") as json_file:
            data_text = json.load(json_file)

    with open(json_imm, errors="ignore") as json_file:
            data_image = json.load(json_file)

    page = 0
    path_image = 'HRDS/images/ACL_2020.acl-main.99/ACL_2020.acl-main.99_' + str(page) +'.jpg'#2022.naacl-industry.16_5.jpg'
    image = Image.open(path_image)
    w, h = image.size
    draw = ImageDraw.Draw(image)

    pdf_path = 'acl_anthology_pdfs_test/2020.acl-main.99.pdf'
    wp, hp = dimension_pdf(pdf_path) #595.276 841.89
    page_text = [item['page'] for item in data_text]# if item['page']==5] 
    f_size = [item['size'] for item in data_text]# if item['page']==5] 
    box_text = [bb_scale(item['box'], w, h, float(wp), float(hp)) for item in data_text]# if item['page']==5]
    for i in data_text:
         i['box'] = bb_scale(i['box'], w, h, float(wp), float(hp)) 

    text = [item['text'] for item in data_text]# if item['page']==5] 

    box_image = [item['box'] for item in data_image]# if int(item['page'])==5] 

    # .. ?'
    new_path = 'check_box_imm_equ/'
    if not os.path.exists(new_path):
         os.makedirs(new_path)

    data = []
    comb = False
    t = 0
    path = 'dataset_parse/image_test/'

    name_im = '2020.acl-main.99'
    print('Stampa')

    path_save = new_path + name_im 
    new_data = []   
   
    new_text_arr = []
    index_im = -1
    first_check = True
    for t in range(len((data_text))):
        check_iou = False
        
        p =int(data_text[t]['page'])
        for i in range(len(data_image)):
             if (p) == int(data_image[i]['page']) \
                and check_overlap(data_text[t]['box'],data_image[i]['box'])>0:
                check_iou = True
                new_text_arr.append({'i': i, 'text':data_text[t]['text']})
                print(new_text_arr[-1])
               # new_text = new_text + data_text[t]['text']
                break #
             
        if check_iou == False: # Non inserisco se ha trovato overlapp
            new_data.append(data_text[t])
        if (t < len(data_text)-1 and p != int(data_text[t+1]['page'])) or t == len(data_text)-1:
            for i in range(len(data_image)):
                  if int(data_image[i]['page']) == p:
                    new_text = ''
                    for m in new_text_arr:
                        print(m)
                        if m['i'] == i:
                             new_text = new_text + ' ' + m['text']
                    
                    print(new_text)
                    dict = {'box': data_image[i]['box'], 'style': False, 'size': False, 'font': False, 'page': int(data_image[i]['page']), 'text': new_text, 'type': data_image[i]['class']}
                    new_data.append(dict)
            new_text_arr = []
  #  print(new_data)
    path_imm= 'HRDS/images/ACL_2020.acl-main.99/ACL_2020.acl-main.99_'
    print(image.size)

    for i in range(len(new_data)-1):
        draw.rectangle(new_data[i]['box'], outline='black' )
        if new_data[i]['page'] != new_data[i+1]['page']:
            image.save(path_save + '_'+ str(new_data[i]['page']) +'.jpg')
            path_image = path_imm + str(new_data[i+1]['page']) +'.jpg'#2022.naacl-industry.16_5.jpg'
            image = Image.open(path_image)
            w, h = image.size
            draw = ImageDraw.Draw(image)
                  

if __name__ == '__main__':
    main()