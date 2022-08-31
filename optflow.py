import numpy as np
import threading
from yolov5.utils.general import (cv2)
import copy
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
            box_img = np.zeros((end_x - start_x, end_y - start_y, 3), dtype=np.int32)
            mask[start_x:end_x, start_y:end_y] = box_img
        cv2.imwrite('mask.jpg', mask)
        return mask
        
    def get_features(self, img, ppbox):
        w, h = img.shape[0:2]
        unit_w = w//self.feature_shape[0]
        unit_h = h//self.feature_shape[1]
        masked_img = self.put_ppbox_mask(img, ppbox)

        pointed_img = copy.deepcopy(img)
        features = [] 

        position = np.array([unit_w, unit_h], dtype=np.int32)
        print(position[0])
        while True:
            if position[0] >= w:
                position[0] = unit_w
                position[1] += unit_h
                continue
            if position[1] >= h:
                break
            if masked_img[position[0], position[1]] is not [0, 0, 0]: 
                pointed_img = cv2.circle(pointed_img, (position[0], position[1]), 10, [0,255,0], -1)
                features.append(position)
            position += np.array([unit_w, 0])
            
        cv2.imwrite('features_points.jpg', pointed_img)
        return features

    def getOpticalFlow(self, prev_img, next_img, prev_features):
        if prev_img is None:
            print('prev_img is none')
            return None
        # do optflow track
        print('prev_feature: ', prev_features)
        gray_prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray_next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
        result_ptr, status, err = cv2.calcOpticalFlowPyrLK(gray_prev_img, gray_next_img, prev_features, None, **self.lk_params)
        return result_ptr[status == 1]
    def draw_optflow(self, frame, pt0, pt1):
        line = np.zeros_like(frame)
        if pt1 is None:
            print('no opt_result!!')
            return
        for old, new in zip(pt0, pt1):
            a,b = new.ravel()
            c,d = old.ravel()
            line = cv2.line(line, (a,b), (c,d), [0,255,255], 1)
            frame = cv2.circle(frame, (a,b), 3, [0,0,255], -1)
            frame = cv2.circle(frame, (c,d), 3, [0,255,0], -1)
        img = cv2.add(frame, line)
        cv2.imwrite('optical_flow.jpg', img)    
            
            
        