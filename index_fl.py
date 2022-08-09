from flask import Flask, render_template, Response, url_for, redirect, request, make_response
from PIL import Image, ImageOps
import cv2
import track
import threading
import time
import os
import json
import urllib.request
import subprocess as sp
app = Flask(__name__)
#camera = cv2.VideoCapture(0)

def legal_url(url):
    try:
        status = urllib.request.urlopen(url).code
        return True
    except Exception as err:
        return False
    
def gen_frames_origin():
    while True:
        time.sleep(0.01)
        frame_origin = open("output.jpg",'rb').read()
        #frame_heatmap = open("output.jpg",'rb').read()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_origin + b'\r\n')
def gen_frames_trace():
    while True:
        time.sleep(0.01)
        frame_origin = open("output_Trace.jpg",'rb').read()
        #frame_heatmap = open("output.jpg",'rb').read()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_origin + b'\r\n')
def gen_frames_flow():
    while True:
        time.sleep(0.01)
        frame_origin = open("output_Arrow.jpg",'rb').read()
        #frame_heatmap = open("output.jpg",'rb').read()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_origin + b'\r\n')
def gen_frames_heatmap():
    while True:
        time.sleep(0.01)
        frame_origin = open("output_heatmap.jpg",'rb').read()
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

@app.route('/')
def start():
    return render_template('start.html')
source = ""
@app.route('/index')
def index():
    global source
    # if start, get source path
    print("before pass: ", source)
    t = threading.Thread(target = track.start_stream, args = (source,))
    t.start()
    time.sleep(1)
    return render_template('index.html')
@app.route('/index', methods=['POST'])
def end_process():
    terminate_t = False
    print("end_process")
    globals.kill_t = True
    return render_template('index.html')
    
    
@app.route('/', methods=['POST'])
def return_home():
    terminate_t = False
    globals.kill_t = True
    print("return_home")
    return render_template('start.html')
    
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
        return render_template('error.html', source = source)

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