import os
import json

def get_info_json(data):
    bounding_boxes = [item['box'] for item in data] # salvo tutte le bb delle miei pagine
    page = [item['page'] for item in data]  
    size = [item['size'] for item in data] 
    text = [item['text'] for item in data]
    type = [item['type'] for item in data]
    style = [item['style'] for item in data]
    return bounding_boxes, page, text, size, type, style

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


def is_number(input_str):
    return input_str.isdigit()

def main(json_file):

    json_file = 'dataset_parse/json/2022.naacl-main.92.json'
    with open(json_file) as f:
        data = json.load(f)
        bounding_boxes, page, text, size, type, style = get_info_json(data)
        labels = []
        edge = []
        for i in range(len(bounding_boxes)): # Ogni bb è un mio nodo
            if style[i] == 'bold': # é titolo/ sec .. 
                if size[i] >= 14 and page[i] == 0: 
                    labels = 0 #  Title 
                elif 11<= size[i] <= 12: # SEC # -> vedi numero prima (?)
                    labels = 1 #  Sec
                elif size[i]<= 10 and page[i] == 0:
                    labels = 2 # author
            elif style[i] == 'italic':
                if size[i] >= 14 and page[i] == 0: 
                    labels = 0 #  Title 
                elif 11<= size[i] <= 12: # SEC # -> vedi numero prima (?)
                    labels = 1 #  Sec
                elif size[i]<= 10 and page[i] == 0:
                    labels = 2 # author
            else:
                if size[i] >= 14 and page[i] == 0: 
                    labels = 0 #  Title 
                elif 11<= size[i] <= 12: # SEC # -> vedi numero prima (?)
                    labels = 1 #  Sec
                elif size[i]< 10 and page[i] == 0:
                    labels = 5 # meta
                elif is_number(text[i]) and bounding_boxes[i][1]> 700: # pagina
                    labels = 6 # Page




if __name__ == '__main__':
    ''

