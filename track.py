
import argparse
from glob import glob
from xml.etree.ElementTree import TreeBuilder
import globals
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import csv
import time
import threading
import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from heatmap import heatmap
from pptrack_handler import PPTrackHandler
from pptracking_util import COLOR_CLOSE, COLOR_LONG, COLOR_MIDDLE, show, BackgroundManager, FlowWorker
from optflow import Optflow
#from curve import draw_trace
import copy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
def video_command():
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'crowdhuman_yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='test_data/testing/0.mp4', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-box', action='store_true', default = True, help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=0,help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--show-heatmap', action='store_true', default=True,help='show heatmap')
    parser.add_argument('--show-arrow', action='store_true', default=True,help='show arrow')
    parser.add_argument('--show-trace', action='store_true', default=True ,help='show trace')
    parser.add_argument('--show-original', action='store_true',help='show original')
    parser.add_argument('--show-optflow', action='store_true',help='show optflow')
    parser.add_argument('--wait', action='store_true', help='when showing img, waiting for user command to continue')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

opt = video_command()
total_heatmap_time = 0
total_trace_time = 0
total_arrow_time = 0
total_optflow_time = 0
n_of_people = 0
people_nums_array = []
heatmap_array = []
trace_array = []
arrow_array = []
yolo_array = []
strongsort_array = []

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])
@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_box=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        show_heatmap = False,
        show_arrow = False,
        show_trace = False,
        show_original = False,
        show_optflow = False,
        wait = False
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = str(yolo_weights).rsplit('/', 1)[-1].split('.')[0]
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = yolo_weights[0].split(".")[0]
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name is not None else exp_name + "_" + str(strong_sort_weights).split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    names = model.module.names if hasattr(model, 'module') else model.names
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_box = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources
    # ---------------------------------------------------------------------------------
    prev_img = None
    prev_crowd_features = None
    ppbox_mask = None
    n_frame = 2 # 決定一次要分析幾個frame , n_frame must>= 2
    pptrack_handler = PPTrackHandler(n_frame)
    b_manager = BackgroundManager()
    first_img = []
    cnt = 0
    min_size = []
    ff=0
    # ---------------------------------------------------------------------------------
    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    
    frame_count_cc = 0
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        if globals.kill_t == True:
            print("STOP ANALYZING!!")
            break
        frame_count_cc += 1
        # if frame_count_cc >= 300:
        #     break
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3
        
        

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0
            tmp_img = copy.deepcopy(im0)
            if im0.shape[0] >1000 or im0.shape[1]>1000:
                tmp_img = cv2.resize(im0,(920,540), interpolation=cv2.INTER_AREA)
            cv2.imwrite("output.jpg", tmp_img)
            ##############################test stream#######################################
            #cv2.imshow(str(p),im0)
            #break
            ################################################################################
            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            background = copy.deepcopy(im0)
            b_manager.refresh(background)

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
            ppl_res = {}  # ppl_res : all ppl's sites, key: id, val: (x,y)
            box_list = {} # box_list : all ppl's box, key: id, val: [start_x, start_y, end_x, end_y]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4]) 
                # ppl_res = xywhs.cpu().numpy()[:, 0:2] # site [the number of people in frame][x center, y center, w, h]
                confs = det[:, 4]
                clss = det[:, 5]                
                
                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        
                        bboxes = output[0:4]
                        id = int(output[4])
                        cls = output[5]
                        bbox_c = np.array([(output[2] + output[0]) // 2, (output[3] + output[1]) // 2]) # center site (x,y)
                        ppl_res[id] = bbox_c
                        box_list[id] = bboxes
                        
                        #############################################find min bbox_size###########
                        w = abs(output[2] - output[0])
                        h = abs(output[3] - output[1])
                        if ff == 0 :
                            min_size = [w, h]
                            ff=1
                        else :
                            if w == 0 or h == 0 :
                                break
                            temp_size = [w, h]
                            if min_size[0]*min_size[1] > temp_size[0]*temp_size[1] :
                                min_size = temp_size
                        #######################################################################
                            # Write MOT compliant results to file
                        
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_box:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            # print("id: ", id,"\n")
                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                    print("min :",min_size,"\nid :",id)
                yolo_array.append(t3-t2)
                strongsort_array.append(t5-t4)
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            
            global total_heatmap_time
            global total_arrow_time
            global total_trace_time
            global total_optflow_time
            globals.n_of_people = len(ppl_res)
            print("People count: ", len(ppl_res))
            
            people_nums_array.append(len(ppl_res))
            
            
           
            
            
            if ppl_res:
                if show_heatmap: 
                    h, w = im0.shape[0:2]
                    heatmap_prev_time = time.time()
                    heatmap(frame_idx, ppl_res, w, h)
                    heatmap_now_time = time.time()
                    temp = heatmap_now_time-heatmap_prev_time
                    total_heatmap_time += temp
                    heatmap_array.append(temp)
                    print("Heatmap_SINGLE_TIME:", temp)
                    #t_heatmap = threading.Thread(target = heatmap(ppl_res, w, h, background))
                    #t_heatmap.start()
                # show arrow diagram(opencv)
                pptrack_handler.add_record(ppl_res)
                if show_arrow or show_trace:   
                    
                    if show_arrow:
                        if len(pptrack_handler.records) >= pptrack_handler.frame_max:
                            edge = int((background.shape[1] + background.shape[0]/2.0)/8.0)
                            
                            arrow_prev_time = time.time()
                            arrow_img = pptrack_handler.draw_crowd_arrow(background, color = COLOR_CLOSE, distance_edge = edge)
                            arrow_now_time = time.time()
                            temp = arrow_now_time-arrow_prev_time
                            total_arrow_time += temp
                            arrow_array.append(temp)
                            print("Arrow_SINGLE_TIME:",temp)
                            show("Arrow", arrow_img, showout = True)
                    if show_trace:
                        if cnt == 0:
                            tmp_h, tmp_w = im0.shape[:2]
                            transparent = np.zeros((tmp_h, tmp_w, 4), dtype = np.uint8)
                        
                            first_img = transparent
                            cnt = 1
                        if len(pptrack_handler.records) >= pptrack_handler.frame_max:
                            pdata = pptrack_handler.trans_data2ppdata(type = 1)
                            trace_prev_time = time.time()
                            curve_img = pptrack_handler.draw_trace(pdata, first_img)
                            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
                            #print(curve_img.shape, background.shape)
                            curve_img = cv2.addWeighted(curve_img, 1, background, 1, 0)
                            trace_now_time = time.time()
                            temp = trace_now_time-trace_prev_time
                            total_trace_time += temp
                            trace_array.append(temp)
                            print("Trace_SINGLE_TIME:", temp)
                            show("Trace", curve_img)
                    if show_optflow:
                        # optflow_prev_time = time.time()
                        optflow = Optflow()     
                        # ppbox_mask= optflow.get_ppbox_mask(im0, box_list)
                        # if prev_features is not None:
                        #     result0, result1 = optflow.get_opticalflow_point(prev_img, im0, prev_features, ppbox_mask)
                        #     optflow.draw_optflow(im0, result0, result1)
                        # prev_img = im0
                        # prev_features = optflow.get_features(im0, ppbox_mask, (100, 100))
                        # optflow_now_time = time.time()
                        # temp = optflow_now_time-optflow_prev_time
                        # total_optflow_time += temp
                        # print("OpticlaFlow_SINGLE_TIME:", temp)
                        if len(pptrack_handler.records) >= pptrack_handler.frame_max:
                            ppbox_mask= optflow.get_ppbox_mask(im0, box_list)  
                            if prev_crowd_features is not None:
                                for i in prev_crowd_features:
                                    crowd_features = prev_crowd_features[i]
                                    result0, result1 = optflow.get_opticalflow_point(prev_img, im0, crowd_features, ppbox_mask)
                                    optflow.draw_optflow(im0, result0, result1)

                            prev_img = im0 # 紀錄上一張圖

                            # 求出上一張圖的features並記錄
                            pdata = pptrack_handler.trans_data2ppdata() 
                            crowd_list= pptrack_handler.get_crowd_list(pdata[0])
                            result = optflow.get_crowds_outer_features_list(im0, ppbox_mask, set(crowd_list), box_list)

                            # 紀錄上一組features
                            prev_crowd_features = result.flatten()
                        time.sleep(1)
                print("TOTAL HEATMAP TIME:", total_heatmap_time)
                print("TOTAL ARROW TIME:", total_arrow_time)
                print("TOTAL TRACE TIME", total_trace_time)
                    
                    
                
            # Stream results
            im0 = annotator.result()
            if show_box:
                box_im = copy.deepcopy(im0)
                box_im = cv2.resize(box_im, (1000,700), interpolation=cv2.INTER_AREA)
                box_im= cv2.putText(box_im, "number of people:"+str(len(ppl_res)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
                show("Box", box_im)

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            if wait:
                print('press "ENTER" to continue')
                input() 
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-box', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--show-heatmap', action='store_true', help='show heatmap')
    parser.add_argument('--show-arrow', action='store_true', help='show arrow')
    parser.add_argument('--show-trace', action='store_true', help='show trace')
    parser.add_argument('--show-original', action='store_true', help='show original')
    parser.add_argument('--wait', action='store_true', help='when showing img, waiting for user command to continue')

    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    global total_heatmap_time
    global total_arrow_time
    global total_trace_time
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    time_start = time.time()
    run(**vars(opt))
    time_end = time.time()
    print("TOTAL OPTICAL FLOW TIME:", total_optflow_time)
    print("TOTAL HEATMAP TIME:", total_heatmap_time)
    print("TOTAL ARROW TIME:", total_arrow_time)
    print("TOTAL TRACE TIME", total_trace_time)
    print("TOTAL TIME:" + format(time_end-time_start))
    # yolo_path = 'yolo_result.csv'
    # yolo_headers = ['yolo_time', 'people']
    # strongsort_path = 'strongsort_result.csv'
    # strongsort_headers = ['strongsort_time', 'people']
    # heatmap_path = 'heatmap_result.csv'
    # heatmap_headers = ['heatmap_time', 'people']
    # arrow_path = 'arrow_result.csv'
    # arrow_headers = ['arrow_time', 'people']
    # flow_path = 'flow_result.csv'
    # flow_headers = ['flow_time', 'people']
    # with open(yolo_path, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     #writer.writerow(yolo_headers)
    #     for i in range(len(yolo_array)):
    #         writer.writerow([yolo_array[i], people_nums_array[i]])
    # with open(strongsort_path, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     #writer.writerow(strongsort_headers)
    #     for i in range(len(strongsort_array)):
    #         writer.writerow([strongsort_array[i], people_nums_array[i]])
    # with open(heatmap_path, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     #writer.writerow(heatmap_headers)
    #     for i in range(len(heatmap_array)):
    #         writer.writerow([heatmap_array[i], people_nums_array[i]])
    # with open(arrow_path, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     #writer.writerow(arrow_headers)
    #     for i in range(len(arrow_array)):
    #         writer.writerow([arrow_array[i], people_nums_array[i]])
    # with open(flow_path, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     #writer.writerow(flow_headers)
    #     for i in range(len(trace_array)):
    #         writer.writerow([trace_array[i], people_nums_array[i]])
#opt = argparse.ArgumentParser().parse_args()


def start_stream(source):
    print("After pass: ", source)
    parser.set_defaults(source = source)
    opt = parser.parse_args()
    main(opt)
    
    
if __name__ == "__main__":
    # opt = video_command()
    main(opt)
    #video_command()