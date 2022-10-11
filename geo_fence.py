import cv2
import copy

dots = []

# class polygon :
#     def __init__(self, dots):
#         self.dots = []  # 點屬性
        
#     def dots_data(self) :
#         for i in len(self.dots) :
#             print(self.dots[i])

polygon_list = []
origin_img = 0

def geofence(origin_img):
    # img = cv2.imread('C:\\Users\\Guan Yu Chen\\Desktop\\project\\0.jpg')
    # img_size = (img.shape[1]//2,img.shape[0]//2)

    # print(img_size)

    # origin_img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    new_img = copy.deepcopy(origin_img)

    cv2.imshow('roi', new_img)
    cv2.setMouseCallback('roi', show_xy)  # 設定偵測事件的函式與視窗

    cv2.waitKey(0)     # 按下任意鍵停止
    cv2.destroyAllWindows()



def show_xy(event,x,y,flags,userdata):
    global new_img
    global origin_img
    #print(event,x,y,flags)
    if event == 1 :                                  # 點擊左鍵
        dots.append([x, y])                          # 記錄座標
        print("start: ",event,x,y,flags)
    elif event == 4 :                                # 放開左鍵
        dots.append([x, y])
        print("end: ",event,x,y,flags)
        cv2.rectangle(new_img, (dots[0][0],dots[0][1]), (dots[1][0],dots[1][1]), (255,255,255), 1)
        dots.clear()
        cv2.imshow('roi', new_img)
    elif event == 5 :                                # 點擊右鍵
        new_img = copy.deepcopy(origin_img)
        cv2.imshow('roi', new_img)
        
    if flags == 1 :                                  # 左鍵拖移
        copy_img = copy.deepcopy(new_img)
        cv2.rectangle(copy_img, (dots[0][0],dots[0][1]), (x,y), (255,255,255), 1)
        cv2.imshow('roi', copy_img)

