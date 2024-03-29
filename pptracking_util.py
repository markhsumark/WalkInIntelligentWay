from re import T
from turtle import back
import numpy as np
import math
from math import sqrt
# from yolov5.utils.general import (cv2)
import cv2
import os
import threading
import copy
import time
import globals

import csv

class ThicknessSigmoid:
    # 控制箭頭粗細的數學式
    def __init__(self, t):
        self.max = t
    def execute(self, n):
        # 實驗出來目前最好看的參數
        x = (n/self.max)* 8 -4
        if x >= 0:
            z = math.exp(-x)
            sig = 1 / (1 + z)
            return sig  * 10
        else:
            z = math.exp(x)
            sig = z / (1 + z)
            return sig * 10

def angle(v1, v2 = [1, 0], tag = True):
    dx1 = v1[0]
    dy1 = v1[1]
    dx2 = v2[0] 
    dy2 = v2[1]
    angle1 = math.atan2(dy1, dx1)
    # arc to degree
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    #if these slope are positive/negative 
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    #if they have different slope
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180 and tag:
            included_angle = 360 - included_angle
    return included_angle
def dist(end, start = np.array([0, 0])):
    vec = end-start
    dis = sqrt(pow(vec[0], 2)+ pow(vec[1], 2))
    return dis

COLOR_CLOSE = (0, 0, 255)
COLOR_MIDDLE = (0, 255, 0)
COLOR_LONG = (255, 0, 0)
#count = 0
def show(title, im, showout = False):
    #global count
    if im.shape[0] >1000 or im.shape[1]>1000:
        w, h = im.shape[0:2]
        im = cv2.resize(im,(1080,int(1080*w/h)), interpolation=cv2.INTER_AREA) 
    try: 
        #print(im.shape)
        # cv2.imwrite("result/output_"+title+str(globals.frame_count_cc)+".jpg", im)
        cv2.imwrite("results/output_"+title+".jpg", im)     
        #count += 1
    except:
        print('---save image of'+ " output_"+title+".jpg" + 'has an error')
    if showout:
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
        vector = self.vector*2
        start = self.start
        thickness = self.thickness*2
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
        if out is None:
            return
        alpha, beta, gamma = 1, self.beta, 0
        self.lock.acquire()
        self.img = cv2.addWeighted(self.img, alpha, out, beta, gamma)
        self.lock.release()
    
def b_search_pp(data_list, f, b, target_id): 
    if b - f <= 0:
        return -1   # not found
    
    m = int((b+ f)/ 2)    
    middle_data = data_list[m]
    if middle_data.id == target_id:
        return m
    else:
        if middle_data.id > target_id:
            return b_search_pp(data_list, f, m, target_id)
        elif middle_data.id < target_id:
            return b_search_pp(data_list, m+1, b, target_id)

class BackgroundManager:
    def __init__(self):
        self.img = []
        self.lock = threading.Lock()
    def refresh(self, new_img):
        self.lock.acquire()
        self.img = new_img
        self.lock.release()
    def get_image(self):
        temp = []
        self.lock.acquire()
        temp = copy.deepcopy(self.img)
        self.lock.release()
        return temp



        
def write_result(path, data_header, data, people_nums_array, write_header = True):
    with open(path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(data_header)
        for i in range(len(data)):
            writer.writerow([data[i], people_nums_array[i]])

def write_all_results(yolo_array = None,
                      strongsort_array= None,
                      heatmap_array= None,
                      arrow_array= None, 
                      trace_array= None,
                      optflow_array = None, 
                      people_nums_array= None
                      ): 
    yolo_path = 'yolo_result.csv'
    yolo_headers = ['yolo_time', 'people']
    strongsort_path = 'strongsort_result.csv'
    strongsort_headers = ['strongsort_time', 'people']
    heatmap_path = 'heatmap_result.csv'
    heatmap_headers = ['heatmap_time', 'people']
    arrow_path = 'arrow_result.csv'
    arrow_headers = ['arrow_time', 'people']
    trace_path = 'flow_result.csv'
    trace_headers = ['flow_time', 'people']
    optflow_path = 'optflow_result.csv'
    optflow_headers = ['optflow_time', 'people']
    if yolo_array:
        write_result(
            path = yolo_path,
            data_header = yolo_headers,
            data = yolo_array, 
            people_nums_array = people_nums_array
        )
    if strongsort_array:
        write_result(
            path = strongsort_path,
            data_header = strongsort_headers,
            data = strongsort_array, 
            people_nums_array = people_nums_array
        )
    if heatmap_array:
        write_result(
            path = heatmap_path,
            data_header = heatmap_headers,
            data = heatmap_array, 
            people_nums_array = people_nums_array
        )
    if arrow_array:
        write_result(
            path = arrow_path,
            data_header = arrow_headers,
            data = arrow_array, 
            people_nums_array = people_nums_array
        )
    if trace_array:
        write_result(
            path = trace_path,
            data_header = trace_headers,
            data = trace_array, 
            people_nums_array = people_nums_array
        )
    if optflow_array:
        write_result(
            path = optflow_path,
            data_header = optflow_headers,
            data = optflow_array, 
            people_nums_array = people_nums_array
        )
