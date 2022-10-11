from ast import Lambda
from faulthandler import disable

from matplotlib.image import imsave
from crowd import Crowd
from pptracking_util import dist, ThicknessSigmoid, color_palette, DrawerManager, angle, b_search_pp
from scipy.spatial.distance import cdist
from collections import deque
from functools import cmp_to_key
import copy
from yolov5.utils.general import (cv2)
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
    def trans_data2ppdata(self, dis_edge = 200, type = 0):
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
                    "nearby":[ data, data, ...]
                    } 
                """
        # find how many people surround each person 
        
        if type == 0 and len(person_data) > 0:
            person_data = self.compute_nearby(person_data, dis_edge)
        pdata_per_frame.append(person_data)
        end = time.time()
        print("- Cost ", end - start, "second in trans_data2ppdata.")
        return pdata_per_frame 
            
    def compute_nearby(self, person_data, edge: int):
        print("classify nearby distance: ", edge)
        dis_array = [x.xy for x in person_data]
        dis_array = cdist(dis_array, dis_array)
        length = len(dis_array)
        limit = length // 2 if length % 2 == 0 else length // 2 + 1
        for row in range(limit):
            for col in range(row + 1, length):
                if dis_array[row][col] <= edge and person_data[row].id != person_data[col].id:
                    person_data[row].nearby.append(person_data[col])
                    person_data[col].nearby.append(person_data[row])
        
        return person_data
    def get_crowd_list(self, pdatas): 
        theta = 30 # define similar direction's included angle
        res_crowd_list = []
        
        # 這段用來找出 附近且走差不多方向 的人
        start = time.time()
        for pp1 in pdatas:
            # 周圍超過1人 
            if len(pp1.nearby) < 1 or int(abs(pp1.vector[0]) + abs(pp1.vector[1])) <= 1:
                pp1.nearby = []
                continue
            new_nearby = [pp1]
            for near_pp in pp1.nearby:
                # 找對應的pid，然後跟據條件合並向量
                for pp2 in pdatas:
                    if pp2.id == near_pp.id:
                        # print("found")
                        vector2 =  pp2.vector
                        vector = pp1.vector
                        
                        # if pp2 isn't move or move slow. (ignore unnessesary people data)
                        if abs(vector2[0]) + abs(vector2[1]) <= 10:
                            break
                        elif angle(vector, vector2) <= theta:
                            new_nearby.append(pp2)
                        break  
            pp1.nearby = sorted(new_nearby, key = cmp_to_key(lambda a, b: a.id - b.id))
        for pp1 in pdatas:
            if len(pp1.nearby) >= 1:
                crowd = Crowd(pp1.nearby)
                res_crowd_list.append(crowd)
        if len(res_crowd_list) == 0:
            print("NO CROWD!!")
            return
        end = time.time()
        # print("- Cost ", end - start, "seconds in 'get_crowd_list()' algo.")
            
        # 去除重複的並加上id
        res_crowd_list = set(res_crowd_list)
        count_id = 1
        for crowd in res_crowd_list:
            crowd.id = count_id
            count_id+= 1

        return res_crowd_list
    def draw_crowd_arrow(self, background, crowd_list, color):
        background = np.array(background, dtype = np.uint8) 
        
        # 取得每個frame中每個人的data
        
        if crowd_list == None:
            return background
        
        largest_crowd = None
        tag = False
        """
        # remove duplicated crowd
        # 去除重複物件的方法: https://minayu.site/2018/12/技術小筆記-利用eq-hash-解決去除重複物件object
        # compare which crowd is the largest if the crowd 
        # find the largest crowd to init the arrow thinkness function 
        """
        for crowd in set(crowd_list):
            if tag == False:
                largest_crowd = crowd 
                tag = True
            elif largest_crowd.size() < crowd.size():
                largest_crowd = crowd 
        arrow_thickness_func = ThicknessSigmoid(largest_crowd.size())
        
        worker_manager = DrawerManager(background, beta = 0.5)        
        for crowd in set(crowd_list):
            worker_manager.add_work(crowd, color, arrow_thickness_func.execute)
            # think_fun trans 4 times to transfor argument to ThicknessSigmoid.execute func
        time1 = time.time()
        worker_manager.work()
        time2 = time.time()
        # print("- Cost: ", time2 - time1,"second in drawing")
        
        background = worker_manager.img
        return background
                
    

        #cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
    def affect_by_optflow(self, person_data, optflow_result): 
        
        for pdata, opt_res_id in zip(person_data, optflow_result):
            optdata = optflow_result[opt_res_id]

            #取opt res的平均位移量
            start_p_list = optdata['start']
            end_p_list = optdata['end']

            total_vector = 0

            vec_list = []

            # test
            print("start_p_list: ", start_p_list)
            print("end_p_list: ", end_p_list)
            vector_list = start_p_list - end_p_list
            print("vector_list: ", vector_list) 
            for point0, point1 in zip(start_p_list, end_p_list):
                if type(point0) is not np.ndarray: 
                    point0 = np.array(point0)
                if type(point1) is not np.ndarray: 
                    point1 = np.array(point1)

                temp_vec = point1 - point0
                vec_list.append(temp_vec)

            # 移除不同的
            # scope_count = np.zeros(8)
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
            
        return person_data





            
                
            




