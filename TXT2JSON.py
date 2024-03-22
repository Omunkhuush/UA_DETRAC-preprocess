from pandas import DataFrame
from xml.etree import ElementTree as ET
from os import listdir
from os.path import isfile, join
import cv2
import json
import numpy as np


def changeId_UA_DETRAC(name):
    cameraId = int(name.split('_')[1])
    frameId = name.split('_')[2].split("img")[1]
    imageId = int(str(cameraId)+str(frameId))
    return imageId

labelsPath = '/projects/UA_DETRAC_dataset/val/labels/'
imagesPath = '/projects/UA_DETRAC_dataset/val/images/'
savePath = '/projects/UA_DETRAC_dataset/val/groundtruth.json'

classes = ['Truck', 'Car', 'Van', 'Bus']

txtFiles = [f for f in listdir(labelsPath) if join(labelsPath, f)]
imageFiles = [f for f in listdir(imagesPath) if join(imagesPath, f)]

info = {}
licenses = {}
categories = []
images = []
annotations = []

for i in range(len(classes)):
    category = {
        'supercategory': classes[i],
        'id': i,
        'name': classes[i]}
    categories.append(category)


for i in range(len(imageFiles)):
    img = cv2.imread(imagesPath+imageFiles[i])
    dh, dw, _ = np.shape(img)
    imgs = {
        'file_name': imageFiles[i],
        'height': dh,
        'width': dw,
        'id': changeId_UA_DETRAC(imageFiles[i].split('.')[0])
    }
    images.append(imgs)


objects = []
k = 0
for i in txtFiles:
    label = open(labelsPath+i,'r')
    label = label.readlines()
    imageName = i.split(".")[0] + ".jpg"
    image_id = changeId_UA_DETRAC(i.split(".")[0])
    img = cv2.imread(imagesPath+imageName)
    height, width, c = np.shape(img)
    for dt in label:
        index, x, y, w, h = map(float, dt.split(' '))
        xmin = int((x - w / 2) * width)
        xmax = int((x + w / 2) * width)
        ymin = int((y - h / 2) * height)
        ymax = int((y + h / 2) * height)
        str(k)
        anns = {
            'id': k+1,
            'image_id': image_id,
            'category_id': int(index),
            'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
            'area': (xmax-xmin)*(ymax-ymin),
            'iscrowd': 0
        }
        annotations.append(anns)
        k += 1

data = {'info': info,
        'licenses': licenses,
        'categories': categories,
        'images': images,
        'annotations': annotations}

with open(savePath, 'w') as json_file:
    json.dump(data, json_file)

print(f"Categories have been saved to {savePath}")
