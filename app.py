from crypt import methods
from flask import Flask, render_template, Response, url_for, redirect, request, make_response, stream_with_context
from PIL import Image, ImageOps
import cv2
import sys
import track
import threading
import time
import os
import json
import urllib.request
import subprocess as sp
import ctypes
import inspect
import globals
from io import BytesIO

app = Flask(__name__)
source = ""
t = threading.Thread(target = track.start_stream, args = (source,))
#t = multiprocessing.Process(target = track.start_stream, args = (source,))
terminate_t = True
#camera = cv2.VideoCapture(0)
def resize_img_2_bytes(image, resize_factor, quality):
    bytes_io = BytesIO()
    img = Image.fromarray(image)

    w, h = img.size
    img.thumbnail((int(w * resize_factor), int(h * resize_factor)))
    img.save(bytes_io, 'jpeg', quality=quality)
    
    return bytes_io.getvalue()

def legal_url(url):
    try:
        status = urllib.request.urlopen(url).code
        return True
    except Exception as err:
        return False
    
def gen_frames_origin():
    while True:
        time.sleep(0.05)
        frame_origin = open("output.jpg",'rb').read()
        #frame_heatmap = open("output.jpg",'rb').read()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_origin + b'\r\n')
def gen_frames_trace():
    while True:
        time.sleep(0.05)
        frame_origin = open("output_Trace.jpg",'rb').read()
        #frame_heatmap = open("output.jpg",'rb').read()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_origin + b'\r\n')
def gen_frames_flow():
    while True:
        time.sleep(0.05)
        frame_origin = open("output_Arrow.jpg",'rb').read()
        #frame_heatmap = open("output.jpg",'rb').read()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_origin + b'\r\n')
def gen_frames_heatmap():
    while True:
        time.sleep(0.05)
        frame_origin = open("output_heatmap.jpg",'rb').read()
        #frame_heatmap = open("output.jpg",'rb').read()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_origin + b'\r\n')
def gen_frames_box():
    while True:
        time.sleep(0.01)
        frame_origin = open("output_Box.jpg",'rb').read()
        #frame_heatmap = open("output.jpg",'rb').read()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_origin + b'\r\n')
# def gen_stream():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             #cv2.imwrite("output.jpg", ret, [cv2.IMWRITE_JPEG_QUALITY, 100])
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_number():
    while True:
        time.sleep(0.01)
        print("-&&&&&---", num)
        num = globals.n_of_people
        yield (b'--text\r\n'
                b'Content-Type: multipart/form-data\r\n\r\n' + str(num) + b'\r\n')

@app.route('/video_origin')
def video_origin():
    return Response(gen_frames_origin(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_trace')
def video_trace():
    return Response(gen_frames_trace(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_flow')
def video_flow():
    return Response(gen_frames_flow(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_heatmap')
def video_heatmap():
    return Response(gen_frames_heatmap(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_box')
def video_box():
    return Response(gen_frames_box(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/number')
# def number():
#     return Response(gen_number(),
#                     mimetype='multipart/x-mixed-replace; boundary=text')

@app.route('/')
def start():
    return render_template('start.html')


@app.route('/index')
def index():
    global source
    #globals.kill_t = True
    time.sleep(1)
    globals.kill_t = False
    t = threading.Thread(target = track.start_stream, args = (source,))
    t.start()
    time.sleep(1)
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def end_process():
    print("end_process")
    globals.kill_t = True
    return render_template('index.html')
    
    
@app.route('/start', methods=['POST'])
def return_home() :
    globals.kill_t = True
    print("return_home")
    return redirect(url_for('start'))


    

    
    

@app.route('/start_process', methods=['POST'])
def start_process_fun():
    global source 
    source = request.form.get('source path')
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    print('is url?: ', is_url)
    if source != "" and (os.path.isfile(source) or (is_url and legal_url(source))):
        if source.startswith('rtmp://'):
            rtmpUrl = source
            camera_path = ""
            cap = cv2.VideoCapture(camera_path)

            # Get video information
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # ffmpeg command
            
            command = ['ffmpeg',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(width, height),
                    '-r', str(fps),
                    '-i', '-',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-f', 'flv', 
                    rtmpUrl]

            # 管道配置
            p = sp.Popen(command, stdin=sp.PIPE)

            # read webcamera
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    print("Opening camera is failed")
                    break
                # process frame
                # your code
                # process frame
                # write to pipe
                p.stdin.write(frame.tostring())
        elif source.startswith('rtsp://'):
            cap = cv2.VideoCapture(source)
            while True:
                # 從 RTSP 串流讀取一張影像
                ret, image = cap.read()
                if not ret:
                    print("Opening camera is failed")
                    break
            # 釋放資源
            cap.release()
        return redirect(url_for('index'))
    else:
        # return error to the user and display on the screen
        # TODO: page error (by mark)
        return render_template('source_error.html', source = source)

# @app.route('/StartInfo/<string:StartInfo>', methods=['GET','POST'])
# def ProcessStartinfo(StartInfo):
#     if request.method == 'POST':
#         StartInfo = json.loads(StartInfo)
#         # check = startinfo['click start']
#         source = StartInfo['source']
#         return redirect(url_for('index'))
        
#         print(source)
    # if source != "" :
    #     return redirect(url_for('index'))
    # else:
    #     # return error to the user and display on the screen
    #     return "<p> the file path cannot be empty!:( </p>"


if __name__ == '__main__':
    app.run('0.0.0.0')
    # app.run('140.121.199.195', port= '5000')