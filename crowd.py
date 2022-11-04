from pptracking_util import dist, Arrow
import numpy as np
class Crowd:
    def __init__(self, people,):
        self.people = tuple(people)
    def __repr__(self):
        return '\n---Crowd---\n人數:{}, ids:{}'.format(len(self.people),[pdata.id for pdata in self.people])
    def __hash__(self):
        return hash(self.__repr__())
    def __eq__(self, other):
        for pp_a in self.people:
            if pp_a not in other.people:
                return False                
        return True
    def size(self):
        return len(self.people)
    def isSame(self, crowd):
        if crowd.people == self.people:
            return True
        else:
            return False
    def get_arrow_mask(self, img, color, thick_fun):
        sum_of_xy = np.zeros(2)
        sum_of_vector = np.zeros(2)
        people = self.people
        for pp in people:
            sum_of_xy = sum_of_xy + pp.xy
            sum_of_vector = sum_of_vector + pp.vector
        start_center_xy = np.array(sum_of_xy / len(people), dtype=np.int)
        if dist(sum_of_vector) <= 2:
            return None
        res_vector = np.array(sum_of_vector, dtype = np.int)
        # thickfun : 用人數去取得箭頭的粗細
        thickness = thick_fun(len(people))
        arrow = Arrow(start_center_xy, res_vector, color, thickness)
        arrow_mask = arrow.get_mask(shape = img.shape)
        return arrow_mask
    