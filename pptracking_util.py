from turtle import back
import numpy as np
import math
from math import sqrt
from yolov5.utils.general import (cv2)
import os
import threading


class ThinknessSigmoid:
    def __init__(self, t):
        self.max = t
    def execute(self, n):
        x = (n/self.max)* 8 -4
        if x >= 0:
            z = math.exp(-x)
            sig = 1 / (1 + z)
            return sig  * 10
        else:
            z = math.exp(x)
            sig = z / (1 + z)
            return sig * 10

def angle(v1, v2):
    dx1 = v1[0]
    dy1 = v1[1]
    dx2 = v2[0] 
    dy2 = v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle
def dist(end, start = np.array([0, 0])):
    vec = end-start
    dis = sqrt(pow(vec[0], 2)+ pow(vec[1], 2))
    return dis

COLOR_CLOSE = (0, 0, 255)
COLOR_MIDDLE = (0, 255, 0)
COLOR_LONG = (255, 0, 0)
        
def show(title, im, wait = 0):
    if im.shape[0] >1000 or im.shape[1]>1000:
        im = cv2.resize(im,(920,540), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, im)
    cv2.waitKey(100)
def color_palette(pp_id:int): 
        x = pp_id % 10
        #rgb -> bgr
        if x==0: res = (0, 255, 255) #Golden Yellow
        elif x==1 : res = (0, 0 ,255) #DarkViolet
        elif x==2: res = (0, 255, 0)#Green
        elif x==3: res = (211, 0 ,148) #red
        elif x==4: res = (255, 0, 0 ) #blue
        elif x==5: res = (130,0,75) #MediumSlateBlue
        elif x==6: res = (19,69,139) #Indigo
        elif x==7: res = (48, 48, 255) #Firebrick1
        elif x==8: res = (113,179,60) #Medium sea green
        else: res = (128, 0, 128) #MediumPurple1
        return res   
class Arrow:
    def __init__(self, start, vector, color, thickness:int):
        self.start = start
        self.vector = vector
        self.color = color
        self.thickness = thickness
    def __repr__(self):
        return '---Arrow---\nstrat_xy:{}\nvec:{}\nthickness:{}\n-----------'.format(self.start, self.vector, self.thickness)
    def get_arrow_points(self, vector, thickness):
        theta = np.radians(90)  #順時鐘轉90度
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        point2 = vector.dot(R)/dist(vector)*thickness
        
        theta = np.radians(-90)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        point3 = vector.dot(R)/dist(vector)*thickness
        
        point1 = point2+ vector/5*3
        point4 = point3+ vector/5*3
        
        point0 = point1+ point2
        point5 = point4+ point3
        return [point0, point1, point2, point3, point4, point5]
    
    def get_mask(self, shape):
        # print(self)
        vector = self.vector
        start = self.start
        thickness = self.thickness
                
        points = self.get_arrow_points(vector, thickness)
        points.append(vector)
        for point in points:
            point +=start
            
        mask = np.zeros((shape), dtype = np.uint8)
        rectangle = np.array(points[1:5], dtype = 'int32')
        triangle = np.array([points[0], points[5], points[6]], dtype = 'int32')
        mask = cv2.fillPoly(mask, [rectangle], COLOR_MIDDLE)
        cv2.fillPoly(mask, [triangle], COLOR_MIDDLE)  
        return mask

    
class DrawerManager:
    def __init__(self, img, beta):
        self.img = img
        self.lock = threading.Lock()
        self.worker_list = []
        self.beta = beta
    
    def add_work(self, obj, color, thick_func):
        worker = threading.Thread(
            target = self.draw_arrow_worker,
            args = (obj, color, thick_func)
            )
        self.worker_list.append(worker)
    def work(self):
        for worker in self.worker_list:
            worker.start()
        for worker in self.worker_list:
            worker.join()
        print("workers done")
    def draw_arrow_worker(self, obj, color, thick_func):
        # 取得旗標
        out = obj.get_arrow_mask(self.img, color, thick_func)
        alpha, beta, gamma = 1, self.beta, 0
        self.lock.acquire()
        self.img = cv2.addWeighted(self.img, alpha, out, beta, gamma)
        self.lock.release()
    
        