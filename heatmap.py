from tracemalloc import start
import numpy as np
from yolov5.utils.general import (cv2)
from numpy import float64, sqrt, tri, vectorize
from numba import jit
import numba
from PIL import Image
import math ,threading, time
from multiprocessing import Pool
from queue import Queue
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import globals
  
def heatmap(frame_idx,ppl_res, width, height):
    if frame_idx % 5 == 0 : #決定一次要分析幾個frame , n_frame must>= 2
        # goal: put the orig frame under the heatmap
        # DEFINE GRID SIZE AND RADIUS(h)
        grid_size = (width + height) // 200
        h = grid_size * 10 # radius which impacts the range
        x = [] # ppl x site
        y = [] # ppl y site
        #ax = []
        for key in ppl_res.keys() :
            x.append(ppl_res[key][0])
            y.append(height - ppl_res[key][1])
            #ax.append((ppl_res[key][0], height - ppl_res[key][1]))

        x_grid = np.arange(0, width, grid_size)
        y_grid = np.arange(0, height, grid_size)
        x_mesh,y_mesh = np.meshgrid(x_grid,y_grid)

        x_c = x_mesh+(grid_size/2)
        y_c = y_mesh+(grid_size/2)
        x_c_len = len(x_c)
        x_c_row_len = len(x_c[0])
        # QUARTIC KERNEL FUNCTION
        
        # print("x_c:",len(x_c), "x_c[0]:", len(x_c[0]), "\n")
        x_c, y_c = np.array(x_c), np.array(y_c)
        x, y = np.array(x), np.array(y)
        #res = []
        
        @jit(nopython=True, nogil=True)
        def cal_int(j,k):
            kde_value, i, x_len  = 0, 0, len(x)
            while i != x_len:
                #d = math.dist([x_c[j][k], y_c[j][k]], [x[i], y[i]])
                d = np.sqrt((x_c[j][k]-x[i])**2+(y_c[j][k]-y[i])**2) #d : the radius in the kernel 
                if d<=h:
                    dn = d/h
                    kde_value += (15/16)*(1-dn**2)**2
                i += 1
            return kde_value
        
        # def cal_k(j, q):
        #     tmp_arr = [cal_int(j, k) for k in range(x_c_row_len)]
        #     q.put(tmp_arr)
        
        # def multithreading():
        #     threads = []
        #     q = Queue()
        #     j = 0
        #     while j < x_c_len:
        #         tmp = 5 if x_c_len-j >= 5 else x_c_len-j
        #         for i in range(tmp):
        #             t = threading.Thread(target=cal_k, args=(i, q))
        #             t.start()
        #             threads.append(t)
        #         for thread in threads:
        #             thread.join()
        #         j += tmp
        #         for _ in range(tmp):
        #             res.append(q.get())
            
        intensity = np.array([[cal_int(j,k) for k in range(x_c_row_len)] for j in range(x_c_len)])
        #multithreading()
        #intensity = np.array(res)
        # print("int_list: ", int_list)
        # print("intensity: ", intensity)
        # print(np.array_equal(int_list, intensity))
        # ---------------------cv2 to pil
        # color_coverted = cv2.cvtColor(background, cv2.COLOR_BGR2RGBA)
        # pil_image = Image.fromarray(color_coverted)       
        #-----------------------plt
    # plt.ion()
    
        fig = plt.figure("heatmap", dpi=150)    

        plt.clf() # clear 
        
        plt.pcolormesh(x_mesh,y_mesh,intensity)
        plt.axis([0, width, 0, height])  
        plt.plot(x,y,'ro')
        plt.savefig('output_heatmap.jpg')
        # plt.savefig('result/output_heatmap'+str(globals.frame_count_cc) +'.jpg')
        #fig = plt.figure()
        #print("fig  : ", type(fig), "\n")
        #-----------------------------------------cv2 to pil---blend  
        # canvas = FigureCanvas(fig)
        #fig2Img = Image.frombytes('RGBA', (width, height), canvas.tostring_rgb()) # image_string
        #blendImg = Image.blend(fig2Img, pil_image, alpha = 0.5)
        #blendImg.show()
        #------------------------------------------pil to cv2
        #fig.canvas.draw()
        # convert canvas to image
        # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
        # img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGRA)
        # img = cv2.resize(width,height)
        # img_blend = cv2.addWeighted(background, 0.5 , img , 0.5, 0)
        # cv2.namedWindow('heatmap')
        # cv2.imshow("heatmap",img_blend)
        # cv2.waitKey()
        # cv2.destroyAllWindows
        #plt.pause(0.01)	# pause 0.01 second
        #plt.ioff() 
        # #return plt
   
