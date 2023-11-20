
# Fare merge con detect
# plot GT 

#??
import os
import json
from PIL import Image, ImageDraw as D


def get_data():
    #immagini
    #labels

    folder_save = 'C:/Users/ninad/Desktop/test_Exp_S/'

    #json per documento
    path_jsons = 'HRDS/test/'
    path_images = 'HRDS/images/'
    list_json = os.listdir(path_jsons)
    labels_set = [] # :
    '''
    {'author', 'alg', 'sec2', 'equ', 'fstline', 'tabcap', 'foot', 'tab', 'fig', 'mail', 'secx', 'title', 
    'sec1', 'figcap', 'para', 'sec3', 'opara', 'fnote', 'affili'}
    '''
    #
    for d in list_json:
        #d = 'EMNLP_D16-1044.json'#'ACL_P10-1160.json'
        path_json = path_jsons + d
        name = d.replace('.json', '')
        folder = path_images + name + '/' 
        draw, image, name_image = get_draw(name, folder, 0) #inizializzo

        with open(path_json, errors="ignore") as json_file:
            data = json.load(json_file)
            
            for i in range(len(data)):
                #print(i.items())
                page = data[i]['page']
                labels_set.append(data[i]['class'])
                #path_iamge = 'ACL_2020.acl-main.99.json'.replace('.json', name)
            # print(path_iamge)

                #passa alla nuova pagina, salvo 
                if i> 0 and page != data[i-1]['page']:  
                    path_save = folder_save +'images_gt/' + name_image
                    #image.save(path_save)
                    draw, image, name_image = get_draw(name, folder, page)
                    
                    print(page)
                if data[i]['class'] == "title":
                    color = 'green'
                elif data[i]['class'] == "sec1":
                    color = 'red'
                elif data[i]['class'] == "sec2":
                    color = 'red'
                elif data[i]['class'] == "sec3":
                    color = 'red'
                elif data[i]['class'] == "tab":
                    color = 'cyan'
                else:
                    color = 'black'
            # elif data[i]['class'] == "para":
            #     color = 'yellow'
                draw.rectangle(data[i]['box'], outline = color) 
        # image.show()
    #print(set(labels_set))

def get_draw(name, folder, page):
    print(folder)
    name_image =name+  '_' + str(page) + '.jpg'
    path_image = folder + name_image
    print(path_image)
    image = Image.open(path_image)
    draw = D.Draw(image)
    return draw, image, name_image

if __name__ == '__main__':
   # parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
    folder = 'exp_test/'
    get_data()
