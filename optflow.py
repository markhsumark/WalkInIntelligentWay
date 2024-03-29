# from msilib.schema import Feature
from multiprocessing.dummy import Process
from turtle import left
from matplotlib.pyplot import box
import numpy as np
import threading
import multiprocessing
from pptrack_handler import PPTrackHandler
from flow_direction import FlowDirection
from yolov5.utils.general import (cv2)
import copy
from pptracking_util import COLOR_CLOSE, COLOR_MIDDLE, dist, show, angle
from collections import deque
from pptrack_handler import Data
from crowd import Crowd
import time

class Optflow: 
    def __init__(self): 
        self.lk_params = dict(  winSize = (15, 15), 
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS| cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_img = None
        self.optflow_result = {}
        self.prev_features = []
    def exec_optical_flow(self, im0, ppbox_list, pdata, draw = False):
        prev_features = self.prev_features
         
        ppbox_mask= self.get_ppbox_mask(im0, ppbox_list)  
        if prev_features is not None:
            optflow_output_img = copy.deepcopy(im0)
            for id in prev_features:
                # 一次處理一個id的
                features = prev_features[id]
                if len(features) !=0:
                    result0, result1 = self.get_opticalflow_point(self.prev_img, im0, features, ppbox_mask)
                    
                    # 存下結果
                    self.optflow_result[id] = {"start": result0, "end": result1}
                    if draw:
                        optflow_output_img = self.draw_optflow(optflow_output_img, result0, result1)
            if draw:
                show('optfolw_result', optflow_output_img, showout = False)
            
            # !!!!!!!!!!!!!!!!optflow result (USE THIS!!!!!!!)
            # print("optflow_result: ", self.optflow_result)
        self.prev_img = im0 # 紀錄上一張圖

        # 求出上一張圖的features並記錄
        # 紀錄上一組features
        prev_feature_time = time.time()
        self.prev_features = self.get_people_outer_features_list(im0, ppbox_mask, pdata[0], ppbox_list) # this part hold most of time of exec_optical_flow
        now_feature_time = time.time()
        print('- Cost {:.3f} second in get_people_outer_features_list'.format(now_feature_time- prev_feature_time))
        
    # 先把人遮住，之後只需判斷是否為mask就能判斷是否在ppbox內, O({人數})
    def get_ppbox_mask(self, img, box_list):  
        img_shape = img.shape
        # 黑色背景
        mask = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.float32)
        for k in box_list:
            box = box_list[k]
            box = box.astype(np.int32)
            start_x, start_y, end_x, end_y = box[0], box[1], box[2], box[3]

            # 在背景蓋上ppbox(指定區域變全白)
            box_img = np.zeros((end_y - start_y, end_x - start_x, 3), dtype=np.int32)
            box_img[0:end_y - start_y, 0:  end_x - start_x] = 255
            mask[start_y:end_y, start_x:end_x] = box_img
        # cv2.imwrite('mask.jpg', mask)
        return mask
        
    
    # check whether in people boxes
    def is_in_ppbox(self, point, mask):
        point = point.astype(np.int32)
        if np.array_equal(mask[point[1], point[0]], np.array([255, 255, 255])): # 白色 -> ppbox 內
            return True
        return False
    # O({crowd數}* {crowd大小}* 100})。回傳每個人群的周圍feature
    def get_people_outer_features_list(self, im0, masked_img, ppdata_list, box_list:dict): 
        img = copy.deepcopy(im0)
        people_outer_features_dict = {}
        
        FDR = Feature_Dict_Runner(
            feature_func = self.get_features,
            result_dict = people_outer_features_dict
            )
        # 處理每個人
        for pp in ppdata_list:
            FDR.add_worker(
                pp = pp,
                box_list= box_list,
                masked_img= masked_img,
                img = img
            )
            # h, w = img.shape[:2]
            # ppl_box = [int(h), int(w), 0, 0]
            # # 找出人的box 外框
            # pid = pp.id
            # ppl_box = np.array(box_list[pid]).astype(np.int32)
            
            # # 避免<0的位置
            # for i in range(4):
            #     if ppl_box[i] < 0 :
            #         ppl_box[i] = 0
            
            # # 取box範圍內的特徵點(相對位置)
            
            # person_outer_features = []
            # while len(person_outer_features) <50: #若<30個點 擴大搜尋範圍
            #     # 外拓，以方便獲取周圍的feature
            #     ppl_box += np.array([-20, -20, 20, 20])
            #     # 取box範圍內的img和masked_img
            #     person_box_masked_img = masked_img[ppl_box[1]:ppl_box[3], ppl_box[0]:ppl_box[2], :]
            #     person_box_origin_img = img[ppl_box[1]:ppl_box[3], ppl_box[0]:ppl_box[2], :]
            #     # prev = time.time()
            #     person_outer_features = self.get_features(person_box_origin_img, person_box_masked_img)# this part Cost a lot of time (~= 0.007 s)
            #     # now = time.time()
            #     # print('get feature Cost :', now - prev)
            
            
            # # 相對位置->絕對位置
            # for feature in person_outer_features:
            #     feature += [ppl_box[0], ppl_box[1]]
            
            # people_outer_features_dict[pp.id] = person_outer_features

        FDR.run()
        people_outer_features_dict = FDR.result_dict
        
        #key: crowd_id, value : features
        return people_outer_features_dict
    # 取人群的輪廓
    def get_features(self, im0, masked_img):
        # 參考來源：https://iter01.com/547012.html
        # ret, binary = cv2.threshold(masked_img, 127, 255, cv2.THRESH_BINARY)
        
        # detect : https://blog.csdn.net/amusi1994/article/details/79591205
        
        # akaze = cv2.AKAZE_create()
        # keypoints = akaze.detect(im0, None)
        
        kaze = cv2.KAZE_create()
        keypoints = kaze.detect(im0, None) # this part cost large time ~= get_features function
        
        # surf = cv2.xfeatures2d.SURF_create()
        # keypoints = surf.detect(im0, None)
        
        # brisk = cv2.BRISK_create()
        # keypoints = brisk.detect(im0, None)
        
        # --------
        all_keypoints = []
        for keypoint in keypoints:
            keypoint = np.array(keypoint.pt, dtype= np.int32)
            if not self.is_in_ppbox(keypoint, masked_img):
                all_keypoints.append(keypoint)
        all_keypoints = np.array(all_keypoints, dtype = np.int32)

        # all_keypoints = np.astype(shape= (1,2), dtype=np.int64)
        
        # masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        # masked_img = np.array(masked_img,np.uint8)
        
        # contours, hierarchy = cv2.findContours(masked_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # draw_img0 = cv2.drawContours(masked_img.copy(), contours, -1,(0,0,255),3)
        
        # contours = np.array(list(contours), dtype = object)
        
        # #trans. data structure to [ [ point coordination ], [ ], ...]
        # res = np.empty(shape= (1,2), dtype=np.int64)
        # for c in contours:
        #     c = c.reshape(c.shape[0]*c.shape[1], 2)
        #     res = np.concatenate((res, c))
        # res = res[1:]
        
        
        
        return all_keypoints

               

    # function of Using optical flow calculatoin and get usable features' movment.
    def get_opticalflow_point(self, prev_img, next_img, prev_features, masked_img):
        if prev_img is None:
            print('(In getOpticalFlow() ): prev_img is none')
            return None
        prev_features = prev_features.reshape(len(prev_features), 1, 2)
        prev_features = prev_features.astype(np.float32)
        # do optflow track
        gray_prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray_next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
        result, status, err = cv2.calcOpticalFlowPyrLK(gray_prev_img, gray_next_img, prev_features, None, **self.lk_params)
        
        all_usable_prev_feature = []
        all_usable_result = []
        prev_features = prev_features.reshape(len(prev_features), 2)
        result = result.reshape(len(result), 2)
        
        # processing that if all the points are in usable position
        for point0, point1, s in zip(prev_features, result, status):

            # 若光流法的"結果"在"圖片內"，則判斷是否不再ppbox內
            if s != 1:
                continue
            # 若光流法的"結果"跑到"圖片外"，則保留資料
            elif point1[1] >= masked_img.shape[0] or point1[0] >= masked_img.shape[1]:
                print("out of mask's bound")
                
            all_usable_prev_feature.append(point0)
            all_usable_result.append(point1)
        return np.array(all_usable_prev_feature), np.array(all_usable_result)
    def draw_optflow(self, frame, pt0s, pt1s):
        if pt1s is None:
            print('no opt_result!!')
            return
        temp_frame = copy.deepcopy(frame)
        pt0s = pt0s.astype(np.int32)
        pt1s = pt1s.astype(np.int32)
        w, h = frame.shape[0: 2]
        limit_vec_dist = dist([w, h]) // 3
        
        # 點線畫法
        for old, new in zip(pt0s, pt1s):
            line_color = [0, 255, 255]
            # remove the strange line
            if dist(new, old) > limit_vec_dist:
                line_color = [0, 0, 0]
            a,b = new.ravel()
            c,d = old.ravel()
            # print("draw line from {} to {}".format(old, new))
            temp_frame = cv2.arrowedLine(temp_frame, (a,b), (c,d), line_color, 7)
            
            temp_frame = cv2.circle(temp_frame, (a,b), 8, [0,0,255], -1)
            temp_frame = cv2.circle(temp_frame, (c,d), 8, [0,255,0], -1)
        
        # 圖像畫法
        people = []
        fake_id_count = 0
        for pt0, pt1 in zip(pt0s, pt1s):
            data = Data(fake_id_count, pt0, pt1-pt0, [])
            fake_id_count+= 1
            people.append(data)
        crowd = Crowd(people)
        f = FlowDirection()
        temp_frame = f.draw_crowd_arrow(temp_frame,[crowd], color = COLOR_MIDDLE)
            # cv2.imwrite('optical_flow.jpg', temp_frame)
        return temp_frame
        
            
            
        
class Feature_Dict_Runner():
    def __init__(self, feature_func, result_dict):
        self.result_dict = result_dict
        self.lock = threading.Lock()
        self.feature_func = feature_func
        self.worker_list = []
    def add_worker(self, pp, box_list, masked_img, img):
        worker = threading.Thread(
            target = self.get_pp_feature,
            args = (pp, box_list, masked_img, img)
        )
        self.worker_list.append(worker)
    def run(self):
        for worker in self.worker_list:
            worker.start()
        for worker in self.worker_list:
            worker.join()
        print('FDRunner workers done!!')
    def get_pp_feature(self, pp, box_list, masked_img, img):
        h, w = img.shape[:2]
        ppl_box = [int(h), int(w), 0, 0]
        # 找出人的box 外框
        pid = pp.id
        ppl_box = np.array(box_list[pid]).astype(np.int32)
        
        # 避免<0的位置
        for i in range(4):
            if ppl_box[i] < 0 :
                ppl_box[i] = 0
        
        # 取box範圍內的特徵點(相對位置)
        
        person_outer_features = []
        while len(person_outer_features) <50: #若<30個點 擴大搜尋範圍
            # 外拓，以方便獲取周圍的feature
            ppl_box += np.array([-20, -20, 20, 20])
            # 取box範圍內的img和masked_img
            person_box_masked_img = masked_img[ppl_box[1]:ppl_box[3], ppl_box[0]:ppl_box[2], :]
            person_box_origin_img = img[ppl_box[1]:ppl_box[3], ppl_box[0]:ppl_box[2], :]
            person_outer_features = self.feature_func(person_box_origin_img, person_box_masked_img)
        
        
        # 相對位置->絕對位置
        for feature in person_outer_features:
            feature += [ppl_box[0], ppl_box[1]]
        with self.lock:
            self.result_dict[pid] = person_outer_features
