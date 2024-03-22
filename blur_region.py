import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import os.path
import multiprocessing

def labelConvert(existLabel,bw,bh):
    boxes = []
    for i in existLabel:
        index, x, y, w, h = map(float, i.split())
        x1 = int((x - w / 2) * bw)
        x2 = int((x + w / 2) * bw)
        y1 = int((y - h / 2) * bh)
        y2 = int((y + h / 2) * bh)
        boxes.append([x1,y1,x2,y2])
    return boxes

def blur_region(img):
    print(img)
    image = cv2.imread(imagesPath+img)
    blured_image = cv2.GaussianBlur(image,(55,55),0)
    bh,bw,channels = image.shape
    cam = "_".join(img.split('_')[:2])
    sorted = [element for element in images if str(cam) in element]
    labels = []
    for k in sorted:
        txtName = k.split(".")[0]+".txt"
        if os.path.exists(labelsPath+txtName):
            f = open(labelsPath+txtName,'r')
            label = f.readlines()
            label = labelConvert(label,bw,bh)
            labels.extend(label)
    pts=[]
    for q in labels:
        x_min, y_min, x_max, y_max = q
        pts.extend([(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)])
    pts_array = np.array(pts, dtype=np.int32)
    mask = np.zeros_like(image)
    hull = cv2.convexHull(pts_array)
    cv2.fillPoly(mask, [hull], color=(255, 255, 255))
    out = np.where(mask == np.array([255, 255, 255]), image, blured_image)
    cv2.imwrite(imagesPath+img,out)

imagesPath = 'val_region_blur/images/'
labelsPath = 'val_region_blur/labels/'

images = [f for f in listdir(imagesPath) if isfile(join(imagesPath,f))]
pool = multiprocessing.Pool()
pool.map(blur_region, images)
pool.close()
pool.join()