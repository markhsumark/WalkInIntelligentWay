from ast import Lambda
from audioop import avg
from faulthandler import disable

from matplotlib.image import imsave
from crowd import Crowd
from pptracking_util import dist, ThicknessSigmoid, color_palette, DrawerManager, angle, b_search_pp, Arrow
from scipy.spatial.distance import cdist
from collections import deque
from functools import cmp_to_key
import copy
# from yolov5.utils.general import (cv2)
import cv2
import numpy as np
import time
import math

class Data:
    def __init__(self, id, xy, vector, nearby):
        self.id = id
        self.xy = xy 
        self.vector = vector
        self.nearby = nearby
    # to check if two objects are same
    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False
    def __repr__(self):
        return 'id:{}, nearby:{}, vector:{}\n'.format(self.id, [n.id for n in self.nearby], self.vector)
    
    
class PPTrackHandler: #Data_position
    def __init__(self, max):
        self.curve_img = []
        self.pdata_per_frame = []
        self.records = deque()
        self.frame_max = max
    
    def draw_trace(self, curve_frame, im):
        for frame in curve_frame:
            for pp_data in frame:
                id = pp_data.id
                vector = pp_data.vector
                start_xy = pp_data.xy
                color = color_palette(id)
                im = cv2.line(im , np.array(start_xy, dtype = int), np.array(start_xy + vector, dtype = int), color, 8) 
        return im  
    """def draw_trace(self):
        background = copy.deepcopy(self.background)
        records = self.records
        for rec in records:
            for person in rec:
               cv2.circle(background, ( rec[person][0],rec[person][1]),  10, color_palette(person), 15) #class
            #person[0] -> x person[1] -> y person[2] -> pp_id
        return background """
    
    def add_record(self, record):
        records = self.records
        records.append(record)
        if len(records) > self.frame_max:
            records.popleft()
    # if no need to nearby data, set type = 1
    def trans_data2ppdata(self):
        start = time.time()
        arrow_record = self.records
        pdata_per_frame = []
        # for i in range(len(arrow_record)-1):
        frame_start = arrow_record[0]
        frame_end = arrow_record[-1]
        person_data = []
        #找出每個pid的start位置以及位移量
        for pid in frame_start:
            if pid in frame_end:
                start_xy = np.array(frame_start[pid])
                dx = frame_end[pid][0] - frame_start[pid][0]
                dy = frame_end[pid][1] - frame_start[pid][1]
                vector = np.array([dx, dy])
                nearby = []
                data = Data(pid, start_xy, vector, nearby)
                person_data.append(data) 
                """
                person_data = [data, data, data ,......]
                data = {
                    "id": 1,  
                    "start_xy": [100, 200], 
                    "vertor" : [20, 50] ,
                    "nearby":[ ] #這裡可能會放Data
                    } 
                """
        # find how many people surround each person 
        
        # if type == 0 and len(person_data) > 0:
        #     person_data = self.compute_nearby(person_data, dis_edge)
        pdata_per_frame.append(person_data)
        end = time.time()
        print("- Cost ", end - start, "second in trans_data2ppdata.")
        return pdata_per_frame 
            
        #cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
        
domain_range = {0:(0, 45), 1:(45, 90), 2:(90, 135), 3:(135, 180), 4:(180, 225), 5:(225, 270), 6:(270, 315), 7:(315, 360)}
def get_target_domain(angle_list):  
    
    domain_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0} #分成８象限
    max_domain =0
    max2_domain =0
    for a in angle_list:
        if a >0 and a<= 45:
            domain_count[0]+= 1
            if domain_count[max_domain] < domain_count[0]:
                max_domain = 0
        elif a> 45 and a <= 90:
            domain_count[1]+= 1
            if domain_count[max_domain] < domain_count[1]:
                max_domain = 1
        elif a> 90 and a <= 135:
            domain_count[2]+= 1
            if domain_count[max_domain] < domain_count[2]:
                max_domain = 2
        elif a> 135 and a <= 180:
            domain_count[3]+= 1
            if domain_count[max_domain] < domain_count[3]:
                max_domain = 3
        elif a> 180 and a <= 225:
            domain_count[4]+= 1
            if domain_count[max_domain] < domain_count[4]:
                max_domain = 4
        elif a> 225 and a <= 270:
            domain_count[5]+= 1
            if domain_count[max_domain] < domain_count[5]:
                max_domain = 5
        elif a> 270 and a <= 315:
            domain_count[6]+= 1
            if domain_count[max_domain] < domain_count[6]:
                max_domain = 6
        elif a> 315 and a <= 360:
            domain_count[7]+= 1
            if domain_count[max_domain] < domain_count[7]:
                max_domain = 7
        
    print('max: ', max_domain)
    return max_domain
def affect_by_optflow(people_data, optflow_result): 
    
    for pdata in people_data:
        
        if pdata.id in optflow_result:
            optdata = optflow_result[pdata.id]
        #取opt res的平均位移量
        start_p_list = optdata['start']
        end_p_list = optdata['end']


        vec_list = end_p_list - start_p_list

        len_vec_list = len(vec_list)
        if len_vec_list == 0:
            continue
        
        ## 利用角度remove extreme vector
        angle_list = [angle(v1 = vec, tag = False) for vec in vec_list]
        # print('angle: ', angle_list)
        
        tgt_d = get_target_domain(angle_list)
        list_in_domain = [[a, v] for a, v in zip(angle_list, vec_list) if a > domain_range[tgt_d][0] and a <= domain_range[tgt_d][1]]
                    
        # math functions
        def average(l):
            l = np.array(l)
            return np.mean(l)
        def variable(l): #變異數
            avg = average(l)
            temp = 0.0
            for i in l:
                temp += math.pow(i - avg, 2)
            return temp/len(l) 
        def sigma(l): #標準差
            var = variable(l)
            return math.pow(var, 0.5) 
        
        # # use average and sigma(標準差)define available range
        temp_list = [a for [a, v] in list_in_domain]
        s = sigma(temp_list)
        avg = average(temp_list)
        available_range = (int(avg-s), int(avg+s))
        # print('range: ', available_range)

        available_vec_list = [v for a, v in list_in_domain if a >= available_range[0] and a <= available_range[1]]
            
        # print('available_vec_list: ', available_vec_list)
            
            

        sum_of_vector = np.zeros(2)
        for vec in available_vec_list:
            sum_of_vector += vec
            # print(vec)
        avg_of_vector = np.array(sum_of_vector / len(available_vec_list), dtype=np.int)   
        # print('avg: ', avg_of_vector)     
        # print('before: ', pdata.vector)
        pdata.vector -= avg_of_vector
        # print('after: ',  pdata.vector)
                
            
    return people_data





            
                
            




