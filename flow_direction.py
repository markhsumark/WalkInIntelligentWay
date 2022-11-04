from scipy.spatial.distance import cdist
import numpy as np
from pptracking_util import dist, ThicknessSigmoid, color_palette, DrawerManager, angle, b_search_pp, show, COLOR_CLOSE
from crowd import Crowd
from pptrack_handler import affect_by_optflow
from functools import cmp_to_key


class FlowDirection:
    def __init__(self):
        self.frame_max = 5
    
    def exec_flow_direction(self, person_data, background, optflow_result = None):
        edge = int((background.shape[1] + background.shape[0]/2.0)/5.0)
        person_data = self.compute_nearby(person_data, edge)
        if optflow_result is not None:
            person_data = affect_by_optflow(person_data, optflow_result)
        res_crowd_list= self.get_crowd_list(person_data)
        arrow_img = self.draw_crowd_arrow(background, res_crowd_list, color = COLOR_CLOSE)
        show("Arrow", arrow_img, showout = True)
        return arrow_img


    def compute_nearby(self, person_data, edge: int):
        # print("classify nearby distance: ", edge)
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
        # start = time.time()
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
        # end = time.time()
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

        worker_manager.work()
        
        background = worker_manager.img
        return background