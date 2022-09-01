import numpy as np
import threading
from yolov5.utils.general import (cv2)
import copy
import time
from pptracking_util import dist
class Optflow: 
    def __init__(self): 
        self.lk_params = dict(  winSize = (15, 15), 
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS| cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
    # 先把人遮住，之後只需判斷是否為mask就能判斷是否在ppbox內
    def get_ppbox_mask(self, img, ppbox):  
        img_shape = img.shape
        # 白色背景
        mask = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.int32)
        mask[0:img_shape[0], 0: img_shape[1]] = 255
        print(len(ppbox[0]))
        for box in ppbox[0]:
            start_x, start_y, end_x, end_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])

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
    
    # get all needed points position in given image
    def get_features(self, img, masked_img, feature_shape):
        w, h = img.shape[0:2]
        unit_w = w//feature_shape[0]
        unit_h = h//feature_shape[1]

        pointed_img = copy.deepcopy(img)
        features = [] 

        # init start position
        position = np.array([unit_h, unit_w], dtype=np.int32)
        while True:
            if position[1] >= w:
                position[0] += unit_h
                position[1] = unit_w
                continue
            if position[0] >= h:
                break
            if not self.is_in_ppbox(position, masked_img): 
                pointed_img = cv2.circle(pointed_img, (position[0], position[1]), 10, [0,255,0], -1)
                features.append(copy.deepcopy(position))
            position += np.array([0, unit_w])
            
        cv2.imwrite('features_points.jpg', pointed_img)
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
        # for pt0 in prev_features:
        result, status, err = cv2.calcOpticalFlowPyrLK(gray_prev_img, gray_next_img, prev_features, None, **self.lk_params)
        
        all_usable_prev_feature = []
        all_usable_result = []
        prev_features = prev_features.reshape(len(prev_features), 2)
        result = result.reshape(len(result), 2)
        for point0, point1, s in zip(prev_features, result, status):
            if s == 1 and not self.is_in_ppbox(point1, masked_img):
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
            # remove the strange point
            if dist(new, old) > limit_vec_dist:
                print("line from {} to {} is strange".format(old, new))
                line_color = [0, 0, 0]
            a,b = new.ravel()
            c,d = old.ravel()
            print("line from {} to {}".format(old, new))
            temp_frame = cv2.arrowedLine(temp_frame, (a,b), (c,d), line_color, 7)
            
            temp_frame = cv2.circle(temp_frame, (a,b), 8, [0,0,255], -1)
            temp_frame = cv2.circle(temp_frame, (c,d), 8, [0,255,0], -1)
            
            # cv2.imwrite('optical_flow.jpg', temp_frame)
            # time.sleep(0.3)
        cv2.imshow('optical_flow', temp_frame)
        cv2.imwrite('optical_flow.jpg', temp_frame)    
            
            
        