import numpy as np
import threading
class Optflow: 
    def __init__(self, feature_shape): 
        self.feature_shape = feature_shape
        self.lk_params = dict(  winSize = (15, 15), 
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS| cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    def put_ppbox_mask(self, img, ppbox):  # 先把人遮住，之後只需判斷是否為mask就能判斷是否在ppbox內
        img_shape = img.shape
        # 白色背景
        mask = np.zeros((img_shape[0], img_shape[1], 3), dtype='uint8')
        mask[0:img_shape[0], 0: img_shape[1]] = '255'
        
        for box in ppbox:
            start_x, width, start_y, height = box[0:4]
            # 在背景蓋上ppbox(指定區域變全黑)
            mask[start_x:width, start_y:height] = (0, 0, 0)
        return mask
        
    def get_features(self, img, ppbox):
        w, h = img.shape[0:2]
        unit_w = w//self.feature_shape[0]
        unit_h = h//self.feature_shape[1]
        masked_img = self.put_ppbox_mask(img, ppbox)

        features = [] 

        position = np.array([unit_w, unit_h])
        
        while True:
            if masked_img[position[0], position[1]] != (0, 0, 0): 
                features.append(position)
            position += np.array([unit_w, 0])
            if position[0] >= w:
                position[0] = unit_w
                position[1] += unit_h
                continue
            if position[1] >= h:
                break

        return features

    def getOpticalFlow(self, prev_img, next_img, prev_features):
        if prev_img is None:
            prev_img = next_img
            return
        # do optflow track
        gray_prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray_next_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        result_ptr, status, err = cv2.calcOpticalFolwPyrLK(gray_prev_img, gray_next_img, prev_features, None, **self.lk_params)
        return result_ptr
