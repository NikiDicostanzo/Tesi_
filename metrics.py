from overlap_box import get_bb_merge, draw_bb, check_overlap
from plot_gt import get_bb_gt, get_name
from PIL import Image, ImageDraw as D

import os

def get_value():
    folder = 'exp_test/'
    path_images = folder + 'images/'

    path_txt = folder + 'labels/' # labels della detection

    path_dataset= './dataset/'
    path_json = path_dataset + "test.json" # GT

    dir_list = os.listdir(path_images) 
    count = 0 
    for d in dir_list:
        #print(dir_list[0])
        path_image = path_images + d
        print('image: ', d)
        #GT -- immagini salvate diversamente nel json
        page, path_name, image_name  = get_name(d)
        bb_gt, labels_gt = get_bb_gt(path_name, path_json, int(page), '') 
        print('GT: ', len(bb_gt), len(labels_gt))
        path_save = folder + 'merge_gt/' + d

        #Detection + merge
        txt = path_txt + image_name + '_' + page +'.txt'
        bb_detect, labels_detect = get_bb_merge(path_image, txt, '')
        print('Detect: ' ,len(bb_detect), len(labels_detect))
        
        image = Image.open(path_image)
        draw = D.Draw(image)
       # if count >= 0:
        remove = []
        tmp = []
        tmp_labels = []
        #devo trovare le bb "fuse"
        for k in range(len(bb_detect)):
            #for i in bb_gt:
            i = 0
            x0 = 1000000
            y0 = 1000000
            x1 = 0
            y1 = 0
        
            merge = False
            while i< len(bb_gt):
                iou = check_overlap(bb_gt[i], bb_detect[k])
                if iou > 0.0:
                    x0 = min(bb_gt[i][0], x0)
                    y0 = min(bb_gt[i][1], y0)
                    x1 = max(bb_gt[i][2], x1) 
                    y1 = max(bb_gt[i][3], y1)

                    remove.append(i) # salvo indice cosi accedo anche a labels
                    lab = labels_gt[i] ## TODO controllo stessa classe
                    merge = True
                    #draw_bb(bb_gt, image, labels_gt, ['red', 'black'], i, draw)
                    #draw_bb(bb_detect, image, labels_detect, ['green','magenta'], k, draw)
                i +=1
            if merge == True:
                
                tmp.append([x0,y0,x1,y1])
                tmp_labels.append(lab)

                merge = False
        
        print(remove)
        remove = sorted(list(set(remove)))
        new_bb = tmp
        new_labes = tmp_labels
        len(tmp)
        for i in range(len(bb_gt)):
            if i not in remove:
                new_bb.append(bb_gt[i])
                new_labes.append(labels_gt[i])
        print(len(new_bb), len(bb_detect))
        draw_bb(new_bb, image, new_labes, ['red', 'black'], k, draw)
        #image.show()
        image.save(path_save)
        #return
    count += 1 




def draw_bb(bb, image, labels, colors, k, draw):
#def draw_bb(bb, path_image, labels, path_save):
    # [x0, y0, x1, y1]
    #image = Image.open(path_image)
    #
    for k in range(len(bb)):
        if labels[k] == 0:
            color = colors[0]
        else:
            color = colors[1]
        draw.rectangle(bb[k], outline = color, width = 1) 
        
    

if __name__ == '__main__':
   # parser = argparse.ArgumentParser(description="...")
   # parser.add_argument("--video", dest="video", default=None, help="Path of the video")
    get_value()


