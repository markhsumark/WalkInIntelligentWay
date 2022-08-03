import numpy as np
from yolov5.utils.general import (cv2)
from numpy import sqrt, tri
from PIL import Image
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#from matplotlib.cbook import get_sample_data
#from matplotlib.backends.backend_gtk4agg import FigureCanvasQTAgg as FigureCanvas
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
  
def heatmap(ppl_res, width, height, background):
    # goal: put the orig frame under the heatmap
    # DEFINE GRID SIZE AND RADIUS(h)
    grid_size = (width + height) // 200
    h = grid_size * 10 # radius which impacts the range
    x = [] # ppl x site
    y = [] # ppl y site
    ax = []
    for key in ppl_res.keys() :
        x.append(ppl_res[key][0])
        y.append(height - ppl_res[key][1])
        ax.append((ppl_res[key][0], height - ppl_res[key][1]))

    x_grid = np.arange(0, width, grid_size)
    y_grid = np.arange(0, height, grid_size)
    x_mesh,y_mesh = np.meshgrid(x_grid,y_grid)
    # print("X_mesh: ", x_mesh, "\n","y_mesh", y_mesh,"datatype: ", type(x_mesh), "\n")
    # print("X_grid: ", x_grid, "\n","y_grid", y_grid, type(x_grid), "\n")

    x_c = x_mesh+(grid_size/2)
    y_c = y_mesh+(grid_size/2)
    # print("x_c datatype", type(x_c))
    #--------------------------
    imagebox = OffsetImage(background, zoom=0.2)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, [0.15,0.5],
                    xybox=(170., -50.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.5
                    )
    #ax.add_artist(ab)
    #--------------------------

    # QUARTIC KERNEL FUNCTION
    def kde_quartic(d,h):
        dn = d/h
        P = (15/16)*(1-dn**2)**2
        return P
    # print("x_c:",len(x_c), "x_c[0]:", len(x_c[0]), "\n")
    int_list=[]
    for j in range(len(x_c)):
        int_row = []
        for k in range(len(x_c[0])):
            #print("k: ", k, "\n")
            kde_value_list = []
            for i in range(len(x)):
                d = math.sqrt((x_c[j][k]-x[i])**2+(y_c[j][k]-y[i])**2) #d : the radius in the kenel 
                if d<=h:
                    # p : density value
                    p = kde_quartic(d,h)
                else:
                    p = 0
                kde_value_list.append(p)
            #print("kde_value_list: ", kde_value_list, "\n")          
            p_total = sum(kde_value_list)
            int_row.append(p_total)
        int_list.append(int_row)
    # ---------------------cv2 to pil
    color_coverted = cv2.cvtColor(background, cv2.COLOR_BGR2RGBA)
    pil_image = Image.fromarray(color_coverted)       
    #-----------------------plt
    plt.ion()
    intensity=np.array(int_list)
   
    fig = plt.figure("heatmap", dpi=120)    
    print("fig datatype : ", type(fig), "\n")
    
    plt.clf() # clear 
    
    plt.pcolormesh(x_mesh,y_mesh,intensity)
    plt.axis([0, width, 0, height])  
    plt.plot(x,y,'ro')
 
    #fig = plt.figure()
    print("fig  : ", type(fig), "\n")
    #-----------------------------------------cv2 to pil---blend  
    canvas = FigureCanvas(fig)
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
    plt.pause(0.01)	# pause 0.01 second
    plt.ioff()
    return plt
