import numpy as np
from os import listdir
from os.path import isfile, join
import cv2 
import multiprocessing
import random

# classes = ["Bus","Bike","Car","Pedestrian","Truck"]

def replace(background, object,new_bbox):
    x1,y1, x2, y2 = new_bbox
    #object = cv2.flip(object, 1)
    roi = background[y1:y1+(y2-y1), x1:x1+(x2-x1)]
    object2gray = cv2.cvtColor(object,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(object2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    background_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    object_fg = cv2.bitwise_and(object,object,mask = mask)
    dst = cv2.add(background_bg,object_fg)
    background[y1:y1+(y2-y1), x1:x1+(x2-x1)] = dst
    return background

def updateLabel(labelPath,category_ID,background,new_bbox):
    f = open(labelPath,"a")
    dh, dw, _=np.shape(background)
    xmin = new_bbox[0]
    ymin = new_bbox[1]
    xmax = new_bbox[2]
    ymax = new_bbox[3]
    x = ((xmin+xmax)/2)/dw
    y = ((ymin+ymax)/2)/dh
    wh = (xmax-xmin)/dw
    he = (ymax-ymin)/dh
    f.write(str(category_ID)+" "+str(x)+" "+str(y)+" "+str(wh)+" "+str(he)+'\n')
    f.close()

def is_overlapping(existing_boxes, new_box):
    #new_box = [new_box[0],new_box[1],new_box[2]-new_box[0],new_box[3]-new_box[1]]
    for box in existing_boxes:
        if (box[0] < new_box[2] and new_box[0] < box[2] and
            box[1] < new_box[3] and new_box[1] < box[3]):
            #print(box,new_bbox)
            return True
    return False

def labelConvert(existLabel,bw,bh):
    boxes = []
    for i in existLabel:
        index, x, y, w, h =i
        x1 = int((x - w / 2) * bw)
        x2 = int((x + w / 2) * bw)
        y1 = int((y - h / 2) * bh)
        y2 = int((y + h / 2) * bh)
        boxes.append([x1,y1,x2,y2])
    return boxes

def get_bbox(objName):
    camera, mode, frame, cls_id, *box = objName.split("_")
    box[-1] = box[-1].split('.')[0]
    x0, y0, x1, y1 = map(int, box[:4])
    new_bbox = [x0,y0,x1,y1]
    return new_bbox

def aug(img):
    classb = ["Pedestrian","Truck","Bus","Car","Bike"]
    image = cv2.imread(imagesPath+img)
    bh,bw,channels = image.shape
    labelPath = labelsPath + img.split(".")[0]+".txt"
    #existLabel = np.loadtxt(labelPath, delimiter=' ', dtype=float)
    cameraMode = "_".join(img.split("_")[:2])
    #random.shuffle(classb)
    for cls in classb:
        sorted_list = [element for element in globals()['category_%s' % cls] if str(cameraMode) in element]
        print(img,cls,len(sorted_list))
        counter = 0
        z = 0
        for obj in sorted_list:
            x = obj.split("_")
            object = cv2.imread(objectPath+cls+'/'+obj)
            if int(x[3]) == 3:
                if z == 2:
                    object = cv2.flip(object, 1)
                    z = 0
                z += 1
            label = np.loadtxt(labelPath, delimiter=' ', dtype=float)
            label = labelConvert(label,bw,bh)
            new_bbox = get_bbox(obj)
            if bw < new_bbox[0] or bw < new_bbox[2] or bh < new_bbox[1] or bh < new_bbox[3]:
                    print('continued!!!')
                    continue             
            overlap = is_overlapping(label,new_bbox)
            if not overlap:
                if counter == augNum:
                    break
                final = replace(image, object, new_bbox)
                cv2.imwrite(imagesPath+img,final)
                updateLabel(labelPath,int(x[3]),image,new_bbox)
                #print(cls,img,obj)
                counter += 1


objectPath = './smallData/smallData_objects/'
imagesPath = './smallData/smallData_10aug_ped_flip/images/'
labelsPath = './smallData/smallData_10aug_ped_flip/labels/'
                
augNum = 10

trainImages = [f for f in listdir(imagesPath) if isfile(join(imagesPath, f))]
classes = ["Bus","Bike","Car","Pedestrian","Truck"]

for cls in classes:
    globals()['category_%s'% cls] = [f for f in listdir(objectPath+cls) if isfile(join(objectPath+cls, f))]
    print(cls,len(globals()['category_%s'% cls]))

pool = multiprocessing.Pool()
pool.map(aug, trainImages)
pool.close()
pool.join()