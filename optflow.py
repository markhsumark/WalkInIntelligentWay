import numpy as np
import threading
from yolov5.utils.general import (cv2)
import copy
import time
from pptracking_util import dist
class Optflow: 
    def __init__(self, feature_shape): 
        self.feature_shape = feature_shape
        self.lk_params = dict(  winSize = (15, 15), 
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS| cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    def put_ppbox_mask(self, img, ppbox):  # 先把人遮住，之後只需判斷是否為mask就能判斷是否在ppbox內
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
        
    def get_features(self, img, ppbox):
        w, h = img.shape[0:2]
        unit_w = w//self.feature_shape[0]
        unit_h = h//self.feature_shape[1]
        masked_img = self.put_ppbox_mask(img, ppbox)

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
            if not np.array_equal(masked_img[position[1], position[0]], np.array([0, 0, 0])): 
                pointed_img = cv2.circle(pointed_img, (position[0], position[1]), 10, [0,255,0], -1)
                # cv2.imwrite('features_points.jpg', pointed_img)
                # time.sleep(0.01)
                features.append(copy.deepcopy(position))
            position += np.array([0, unit_w])
            
        cv2.imwrite('features_points.jpg', pointed_img)
        return np.array(features)

    def getOpticalFlow(self, prev_img, next_img, prev_features):
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
        # all_corresponded_feature.append(result[status == 1])
        print(result[status == 1])
        return result[status == 1]
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
            temp_frame = cv2.line(temp_frame, (a,b), (c,d), line_color, 5)
            temp_frame = cv2.circle(temp_frame, (a,b), 10, [0,0,255], -1)
            temp_frame = cv2.circle(temp_frame, (c,d), 10, [0,255,0], -1)
            cv2.imwrite('optical_flow.jpg', temp_frame)
            time.sleep(0.3)
        cv2.imwrite('optical_flow.jpg', temp_frame)    
            
            
        