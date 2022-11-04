from ast import Lambda
from faulthandler import disable

from matplotlib.image import imsave
from crowd import Crowd
from pptracking_util import dist, ThicknessSigmoid, color_palette, DrawerManager, angle, b_search_pp
from scipy.spatial.distance import cdist
from collections import deque
from functools import cmp_to_key
import copy
# from yolov5.utils.general import (cv2)
import cv2
import numpy as np
import time

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
def affect_by_optflow(person_data, optflow_result): 
    
    for pdata, opt_res_id in zip(person_data, optflow_result):
        optdata = optflow_result[opt_res_id]

        #取opt res的平均位移量
        start_p_list = optdata['start']
        end_p_list = optdata['end']

        total_vector = 0

        # vec_list = []

        # test
        vec_list = end_p_list - start_p_list

        # 移除不合理的向量
        # scope_count = np.zeros(8) # 八個方向
        # for vec in vec_list:
        #     angle = angle(vec, [1, 0])
        #     scope = angle/45
        #     scope_count[scope]+= 1
        #...未完成
        if len(vec_list)!= 0:
            for vec in vec_list:
                total_vector += vec
            avg_vector = total_vector/len(vec_list)
            pdata.vector -= avg_vector
        
        for pdata, opt_res_id in zip(person_data, optflow_result):
            optdata = optflow_result[opt_res_id]

            #取opt res的平均位移量
            start_p_list = optdata['start']
            end_p_list = optdata['end']

            total_vector = 0

            # vec_list = []

            # test
            vec_list = end_p_list - start_p_list

            # 移除不合理的向量
            # scope_count = np.zeros(8) # 八個方向
            # for vec in vec_list:
            #     angle = angle(vec, [1, 0])
            #     scope = angle/45
            #     scope_count[scope]+= 1
            # for i in range(8):
                
            #...未完成
            if len(vec_list)!= 0:
                for vec in vec_list:
                    total_vector += vec
                avg_vector = total_vector/len(vec_list)
                pdata.vector -= avg_vector
            
        return person_data





            
                
            




