from turtle import left
import numpy as np
import threading
from yolov5.utils.general import (cv2)
import copy
import time
from pptracking_util import dist, show
from collections import deque

class Optflow: 
    def __init__(self): 
        self.lk_params = dict(  winSize = (15, 15), 
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS| cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
    # 先把人遮住，之後只需判斷是否為mask就能判斷是否在ppbox內, O({人數})
    def get_ppbox_mask(self, img, box_list):  
        img_shape = img.shape
        # 白色背景
        mask = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.int32)
        mask[0:img_shape[0], 0: img_shape[1]] = 255
        for k in box_list:
            box = box_list[k]
            start_x, start_y, end_x, end_y = box[0], box[1], box[2], box[3]

            # 在背景蓋上ppbox(指定區域變全黑)
            box_img = np.zeros((end_y - start_y, end_x - start_x, 3), dtype=np.int32)
            mask[start_y:end_y, start_x:end_x] = box_img
        cv2.imwrite('mask.jpg', mask)
        return mask
        
    
    # check whether in people boxes
    def is_in_ppbox(self, point, mask):
        point = point.astype(np.int32)
        if np.array_equal(mask[point[1], point[0]], np.array([0, 0, 0])): 
            return True
        return False
    # O({crowd數}* {crowd大小}* 100})
    def get_crowds_outer_features_list(self, im0, masked_img, crowd_list, box_list:dict): 
        img = copy.deepcopy(im0)
        h, w = img.shape[:2]
        count = 0
        crowds_outer_features_list = []
        # 處理每個人群
        for crowd in crowd_list:
            crowd_box = [h, w, 0, 0]
            # 找出人群的box 外框
            for ppl_data in crowd: 
                pid = ppl_data.id
                ppl_box = box_list[pid]
                start_x, start_y, end_x, end_y = ppl_box[0], ppl_box[1], ppl_box[2], ppl_box[3]
                if start_x < crowd_box[0]:
                    crowd_box[0] = start_x
                if start_y < crowd_box[1]:
                    crowd_box[1] = start_y
                if end_x > crowd_box[2]:
                    crowd_box[2] = end_x
                if end_y > crowd_box[3]:
                    crowd_box[3] = end_y
            # 取box範圍內的img和masked_img
            crowd_box_img = img[crowd_box[0]:crowd_box[2], crowd_box[1]:crowd_box[3]]
            crowd_box_masked_img = masked_img[crowd_box[0]:crowd_box[2], crowd_box[1]:crowd_box[3], :]
            cv2.imwrite('crowd_box({})'.format(count), crowd_box_img)
            # 取box範圍內的特徵點
            crowd_outer_features = self.get_features(crowd_box_img, crowd_box_masked_img)
            crowds_outer_features_list.append(crowd_outer_features)

            count += 1
        cv2.imwrite('crowds_outer_features_result.jpg', img)
        return np.array(crowds_outer_features_list, dtype = np.int32)
    # get all needed points position in given image
    def get_features(self, img, masked_img, feature_shape = (10, 10)):
        # states = np.zeros(feature_shape, dtype = np.bool)

        h, w = img.shape[0:2]
        unit_h = h//feature_shape[0]
        unit_w = w//feature_shape[1]

        pointed_img = copy.deepcopy(img)
        features = [] 
        position = np.array([unit_w, unit_h], dtype=np.int32)
        # 先找出人周圍的點（包含在人的框框內的）
        while True:
            if position[1] >= h:
                position[0] += unit_w
                position[1] = unit_h
                continue
            if position[0] >= w:
                break

            if not self.is_in_ppbox(position, masked_img):
                pointed_img = cv2.circle(pointed_img, (position[0], position[1]), 10, [0,255,0], -1)
                features.append(position)
            position += np.array([0, unit_h]) 
        return np.array(features)

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
            # 若光流法的結果跑到"圖片外"，則保留資料
            if point1[1] >= masked_img.shape[0] or point1[0] >= masked_img.shape[1]:
                print("out of mask's bound")
            # 若光流法的結果在"圖片內"，則判斷是否不再ppbox內
            elif s != 1 or self.is_in_ppbox(point1, masked_img):
                continue
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
        show('optical_flow', temp_frame, showout = False)
        cv2.imwrite('optical_flow.jpg', temp_frame)    
            
            
        