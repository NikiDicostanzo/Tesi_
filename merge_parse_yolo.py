import os 
import json
from PIL import Image, ImageDraw

from graph_parse import get_font_comm
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

def is_inside(boxIn, boxEx):
    return boxEx[0]<= boxIn[0] and boxEx[1]<= boxIn[1] and boxEx[2]>= boxIn[2] and boxEx[3]>= boxIn[3]

def check_box_inside(box_ex, box2_in):
      return box_ex[0] <= box2_in[0] and box_ex[1] <= box2_in[1] \
        and box_ex[2] >= box2_in[2] and box_ex[3] >= box2_in[1]

# yolo_hrdhs_3/jons_yolo/
# dataset_parse/ json
def main():
    #json_name = 'D13-1134.json'#'2022.naacl-industry.16.json'
    path_json_text = 'yolo_hrds_4_gt_test/json_parse/'#'yolo_hrdhs_672_3_testGT2/json_parse/' #+ json_name

    path_json_imm = 'yolo_hrds_4_gt_test/json_yolo/'#EMNLP_D13-1134.json'

    save_path ='yolo_hrds_4_gt_test/json/'
    if not os.path.exists(save_path):
         os.makedirs(save_path)
    list_json = os.listdir(path_json_imm)
    for j in list_json:
        print(j)
        json_imm = path_json_imm + j

        json_name = '_'.join(j.split('_')[1:])
        all_name = j.replace('.json', '')
        json_text = path_json_text + json_name
        name = json_name.replace('.json', '')
    
        #'yolo_hrdhs_672_3_testGT2/json_yolo/ACL_2020.acl-main.99.json'# + json_name
        with open(json_text, errors="ignore") as json_file:
                data_text = json.load(json_file)
        with open(json_imm, errors="ignore") as json_file:
                data_yolo = json.load(json_file)

        f = [item['font'] for item in data_text]
        comm_font = get_font_comm(f)
        page = 0
        path_image = 'HRDS/images/'+ all_name +'/' + all_name + '_' + str(page) +'.jpg'#2022.naacl-industry.16_5.jpg'
        image = Image.open(path_image)
        w, h = image.size
        draw = ImageDraw.Draw(image)

        pdf_path = 'acl_anthology_pdfs_test/' + name + '.pdf'
        wp, hp = dimension_pdf(pdf_path) #595.276 841.89
        for i in data_text:
            i['box'] = bb_scale(i['box'], w, h, float(wp), float(hp)) 

        new_path = 'TESI_check_box_imm_equ/'
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        
        new_data = get_dict_merge(data_text, data_yolo, comm_font)

        save_json = save_path + j 
        with open(save_json, 'w') as f:
            json.dump(new_data, f, indent=4)

        
    
    #     name_im = 'D13-1134'
    #     path_save = new_path + name_im 
    #     path_imm= 'HRDS/images/EMNLP_D13-1134/EMNLP_D13-1134_'

    #     for i in range(len(new_data)-1):
    #         color = get_color(new_data[i]['type'])
    #         draw.rectangle(new_data[i]['box'], outline=color )
    #         if new_data[i]['page'] != new_data[i+1]['page']:
    #             image.save(path_save + '_'+ str(new_data[i]['page']) +'.jpg')
    #             path_image = path_imm + str(new_data[i+1]['page']) +'.jpg'#2022.naacl-industry.16_5.jpg'
    #             image = Image.open(path_image)
    #             w, h = image.size
    #             draw = ImageDraw.Draw(image)

def get_dict_merge(data_text, data_image, comm_font):
    t = 0
    new_data = []   
    new_text_arr = []
    for t in range(len((data_text))):
        check_iou = False
        
        p =int(data_text[t]['page'])
        for i in range(len(data_image)):
            if (p) == int(data_image[i]['page']) \
                and (check_overlap(data_text[t]['box'],data_image[i]['box'])>0):# and comm_font not in data_text[t]['font'])\
                     #or  is_inside(data_text[t]['box'],data_image[i]['box']):
                t1 = data_text[t]['text']
                # nel caso in cui la box della tab o imm è piu grande e prende la caption 
                if (('Table' in t1 or 'Figure' in t1) and (':' in t1 or len(t1.replace(' ', ''))<9)) and comm_font in data_text[t]['font']:
                     break
                else:
                    check_iou = True
                    new_text_arr.append({'i': i, 'text':data_text[t]['text'], 'block':data_text[t]['block']})
                    break 
             
        if check_iou == False: # Non inserisco se ha trovato overlapp
            new_data.append(data_text[t])
        if (t < len(data_text)-1 and p != int(data_text[t+1]['page'])) or t == len(data_text)-1:
            for i in range(len(data_image)):
                  if int(data_image[i]['page']) == p:
                    new_text = ''
                    block = 0
                    for m in new_text_arr:
                        #print(m)
                        if m['i'] == i:
                             block = m['block']
                             new_text = new_text + ' ' + m['text']
                    
                    #print(new_text)
                    dict = {'block': block, 'box': data_image[i]['box'], 'style': False, 'size': False, 'font': False, 'page': int(data_image[i]['page']), 'text': new_text, 'type': data_image[i]['class']}
                    new_data.append(dict)
            new_text_arr = []
    return new_data
                  

if __name__ == '__main__':
    main()