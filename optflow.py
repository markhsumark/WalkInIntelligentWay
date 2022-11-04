from turtle import left
import numpy as np
import threading
from yolov5.utils.general import (cv2)
import copy
from pptracking_util import dist, show, angle
from collections import deque

class Optflow: 
    def __init__(self): 
        self.lk_params = dict(  winSize = (15, 15), 
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS| cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
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
        h, w = img.shape[:2]
        people_outer_features_dict = {}
        # 處理每個人群
        for pp in ppdata_list:
            
            ppl_box = [int(h), int(w), 0, 0]
            # 找出人的box 外框
            pid = pp.id
            ppl_box = np.array(box_list[pid]).astype(np.int32)
            # 外拓，以方便獲取周圍的feature
            ppl_box += np.array([-10, -10, 10, 10])
            # 避免<0的位置
            for i in range(4):
                if ppl_box[i] < 0 :
                    ppl_box[i] = 0
            # 取box範圍內的img和masked_img
            person_box_masked_img = masked_img[ppl_box[1]:ppl_box[3], ppl_box[0]:ppl_box[2], :]
            # 取box範圍內的特徵點(相對位置)
            person_outer_features = self.get_features(person_box_masked_img)
            
            # 相對位置->絕對位置
            for feature in person_outer_features:
                feature += [ppl_box[0], ppl_box[1]]
                
                img = cv2.circle(img, (feature[0], feature[1]), 10, [0,255,0], -1)
            
            people_outer_features_dict[pp.id] = person_outer_features

        cv2.imwrite('person"s_all_features.jpg', img)

        #key: crowd_id, value : features
        return people_outer_features_dict
    # 取人群的輪廓（最多x個點作為特徵)
    def get_features(self, masked_img):
        # 參考來源：https://iter01.com/547012.html
        # ret, binary = cv2.threshold(masked_img, 127, 255, cv2.THRESH_BINARY)
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        masked_img = np.array(masked_img,np.uint8)
        contours, hierarchy = cv2.findContours(masked_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # draw_img0 = cv2.drawContours(masked_img.copy(), contours, -1,(0,0,255),3)
        
        contours = np.array(list(contours), dtype = object)
        
        #trans. data structure to [ [ point coordination ], [ ], ...]
        res = np.empty(shape= (1,2), dtype=np.int64)
        for c in contours:
            c = c.reshape(c.shape[0]*c.shape[1], 2)
            res = np.concatenate((res, c))
        res = res[1:]
        
        return res

               

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
            
            # cv2.imwrite('optical_flow.jpg', temp_frame)
        return temp_frame
        
            
            
        