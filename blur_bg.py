import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import os.path
import math 
imagesPath = 'smallData/images/'
labelsPath = 'smallData/labels/'
#classes = ['Bus', 'Bike', 'Car', 'Pedestrian', 'Truck']

def calculate_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def is_point_outside_line(p, line_start, line_end):
    """
    Check if point p is outside the line segment formed by line_start and line_end.
    """
    # Calculate distances
    dist_point_to_start = calculate_distance(p, line_start)
    dist_point_to_end = calculate_distance(p, line_end)
    dist_start_to_end = calculate_distance(line_start, line_end)
    
    # Check if the point is outside the line segment
    if dist_point_to_start + dist_point_to_end > dist_start_to_end:
        return True  # The point is outside the line segment
    else:
        return False  # The point is on or within the line segment

def labelConvert(existLabel,bw,bh):
    boxes = []
    for i in existLabel:
        # print("i= ",i)
        # print(i.split(" "))
        index, x, y, w, h = map(float, i.split())
        x1 = int((x - w / 2) * bw)
        x2 = int((x + w / 2) * bw)
        y1 = int((y - h / 2) * bh)
        y2 = int((y + h / 2) * bh)
        boxes.append([x1,y1,x2,y2])
    return boxes
def find_cord(labels):
    # xmin = min(labels, key=lambda x: x[0])
    # ymin = min(labels, key=lambda x: x[1])
    # xmax = max(labels, key=lambda x: x[2])
    # ymax = max(labels, key=lambda x: x[3])
    # pts = [(xmin[0],xmin[3]),(xmin[0],xmin[1]),
    #        (ymin[0],ymin[1]),(ymin[2],ymin[1]),
    #        (xmax[2],xmax[1]),(xmax[2],xmax[3]),
    #        (ymax[2],ymax[3]),(ymax[0],ymax[3])]
    pts=[]
    for i in labels:
        x_min, y_min, x_max, y_max = i
        pts.extend([(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)])
    return pts


images = [f for f in listdir(imagesPath) if isfile(join(imagesPath,f))]
k = images[0:20]
for i in k:
    print(i)
    image = cv2.imread(imagesPath+i)
    blured_image = cv2.GaussianBlur(image,(55,55),0)
    bh,bw,channels = image.shape
    txtName = i.split(".")[0]+'.txt'
    f = open(labelsPath+txtName,'r')
    label = f.readlines()
    label = labelConvert(label,bw,bh)
    print(label)
    for dt in label:
        x_min, y_min, x_max, y_max = dt
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
    pts = find_cord(label)
    pts_array = np.array(pts, dtype=np.int32)
    #pts_array = pts_array.reshape((-1, 1, 2))
    #pts_array = pts_array.reshape((-1, 1, 2))
    mask = np.zeros_like(image)
    hull = cv2.convexHull(pts_array)
    #cv2.rectangle(mask, (cords[0], cords[1]), (cords[2], cords[3]), (255, 255, 255), -1)
    #cv2.polylines(mask, [pts_array], isClosed=True, color=(255, 255, 255),thickness=50)
    cv2.fillPoly(mask, [hull], color=(255, 255, 255))

    out = np.where(mask == np.array([255, 255, 255]), image, blured_image)
    #cv2.rectangle(out, (cords[0], cords[1]), (cords[2], cords[3]), (0,0,255), 3)
    cv2.polylines(out, [hull], isClosed=True, color=(0, 0, 255), thickness= 2)
    cv2.imwrite('exp2_blur/'+i,out)
