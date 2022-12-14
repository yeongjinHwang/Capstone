import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import imutils
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import sys
from random import randint
import select
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import math
import serial
from arduino import arduino2
import webbrowser   # add

####GPU로 쓸게요####
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

####입력인자로 --XXXX 매크로####
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video1', '2', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('video2', '4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')

def main(_argv):
    ####최초 init####
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video1
    video_path2 = FLAGS.video2
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
     ### add
    url = 'file:///home/yeongjin/yoloV4DeepSort/yolov4-deepsort/payment.html'
    url2 = 'https://www.google.com/'
    queryStr = '?mid=53333&'
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    payflag = [0 for _ in range(100)]

    ####비디오 캡쳐####
    face_cap = cv2.VideoCapture(4) # Face Cam # add
    vid = cv2.VideoCapture(int(video_path))
    vid2 = cv2.VideoCapture(int(video_path2))
    out = None

    ####계산대 좌표 받아오기####
    firstReturn, firstFrame = vid.read()
    firstReturn2, firstFrame2 = vid2.read()
    firstFrame = cv2.flip(firstFrame,1)
    firstFrame2 = cv2.flip(firstFrame2,1)
    resultPos = cv2.hconcat([firstFrame, firstFrame2])
    print('boxing INPUT XY')
    xName, yName, nameWidth, nameHeight = cv2.selectROI("location",resultPos,False)
    cv2.destroyAllWindows()  
    print('boxing a cash register1')
    xCal, yCal, calWidth, calHeight = cv2.selectROI("location",resultPos,False)
    cv2.destroyAllWindows()
    print('boxing a cash register2')
    x2Cal, y2Cal, calWidth2, calHeight2 = cv2.selectROI("location",resultPos,False)
    cv2.destroyAllWindows()
    print('overLap')
    xOver, yOver, overWidth, overHeight = cv2.selectROI("location",resultPos,False)
    cv2.destroyAllWindows()
    print('QR XY')  # add
    xQR, yQR, qrWidth, qrHeight = cv2.selectROI("location",resultPos,False)
    cv2.destroyAllWindows()
    namingX, namingY= xName+nameWidth/2, yName+nameHeight/2
    medianX, medianY, medianX2, medianY2 = xCal+calWidth/2, yCal+calHeight/2, x2Cal+calWidth2/2, y2Cal+calHeight2/2
    print("naming location : x "+str(xName)+" y "+str(yName)+" width "+str(nameWidth)+" height "+str(nameHeight)+\
    " namingX : "+str(namingX)+' namingY : '+str(namingY))
    print("cal1 location : x "+str(xCal)+" y "+str(yCal)+" width "+str(calWidth)+" height "+str(calHeight)+\
    " medianX : "+str(medianX)+' medianY : '+str(medianY))
    print("cal2 location : x "+str(x2Cal)+" y "+str(y2Cal)+" width "+str(calWidth2)+" height "+str(calHeight2)+\
    " medianX : "+str(medianX2)+' medianY : '+str(medianY2))
    print("overLap location : x "+str(xOver)+"~"+str(xOver+overWidth)+" y "+str(yOver)+"~"+str(yOver+overHeight))

    ####--output path 인자로 시작하면 저장하기 위한 코드####
    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    ####영상or웹캠실행####
    frameDrop,prevNameLen,curNameLen=0,0,0
    nameBuf, nametoTrackId, distName, dist_1, dist_2 = [],[],[],[],[]

    for i in range(100):
        nametoTrackId.append(i)
        distName.append(math.inf)
        dist_1.append(math.inf)
        dist_2.append(math.inf)
    ### arduino
    purchase = [['none'] for _ in range(50)]
    purchase2 = [['none'] for _ in range(50)] 
    userBuy_1 = 0
    userBuy_2 = 0
    userBuy_list_1 = []
    userBuy_list_2 = []
    ### html add
    product = ['shampoo', 'body wash', 'cleansing foam']
    product2 = ['coffee', 'rice', 'tissue']
    product_price = [12900, 11000, 8900]
    product2_price = [2500, 1100, 500]
    arduino_ = serial.Serial(port = "/dev/ttyACM0", baudrate = 115200)
    while True:
        frameDrop=frameDrop+1
        if frameDrop%2==0 :
            ###async input name
            prevNameLen=len(nameBuf)
            while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                name = sys.stdin.readline()
                if name:
                    nameBuf.append(name[:-1])
            curNameLen=len(nameBuf)

            ####프레임 받아오고 FLIP으로 좌우반전####
            return_value, frame = vid.read()
            return_value2, frame2 = vid2.read()
            frame = cv2.flip(frame,1)
            frame2 = cv2.flip(frame2,1)
            frame = cv2.hconcat([frame, frame2])
            ####opencv는 color를 bgr 방식으로 저장하는데, 이를 rgb방식으로 변환####
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ####넘파이로 저장된 이미지 배열을 PIL 이미지로 변환####
            image = Image.fromarray(frame)

            ####SIZE변경####
            frame_size = (frame.shape[:2])
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            start_time = time.time() ##FPS 구해서 출력하려고
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            pred_bbox = [bboxes, scores, classes, num_objects]
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)
            allowed_classes = ['person']

            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            # count = len(names)

            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                bbox = track.to_tlbr()
                x=int((bbox[0]+bbox[2])/2)
                y=int((bbox[1]+bbox[3])/2)
                distName[track.track_id] = (x-namingX)**2 + (y-namingY)**2
                dist_1[track.track_id] = (x-medianX)**2 + (y-medianY)**2
                dist_2[track.track_id] = (x-medianX2)**2 + (y-medianY2)**2

                if prevNameLen != curNameLen: 
                    nametoTrackId[distName.index(min(distName))] = nameBuf[len(nameBuf)-1]
                if x > xQR and x < (xQR+qrWidth) and y > yQR and y < (yQR+qrHeight): #add
                    if purchase[track.track_id] == ['none'] and purchase2[track.track_id] == ['none'] :
                        continue
                    #ret, frame1 = face_cap.read() 
                    #gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    #faces = faceCascade.detectMultiScale(
                    #     gray, # 원본
                    #     scaleFactor = 1.2, # 검색 윈도우 확대 비율, 1보다 커야 한다
                    #     minNeighbors = 6, #얼굴 사이 최소 간격 (픽셀)
                    #     minSize=(20,20) # 얼굴 최소 크기 (보다 작으면 무시)
                    # )

                    # 얼굴에 대해 rectangle 출력
                    #for (x,y,w,h) in faces:
                    #    cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    if payflag[track.track_id] == 0 : 
                        #cv2.putText(frame1, 'Face Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (254, 1, 15), 2, cv2.LINE_AA)
                    # QR payment
                        mid_int = 53800 # merchandise number
                        mid_int += 1
                        mid_str = str(mid_int)
                        mname_str = ''
                        if purchase[track.track_id] != 'none' :
                            for i in purchase[track.track_id]:
                                mname_str += i + ' '
                        if purchase[track.track_id] != 'none' :
                            for i in purchase2[track.track_id]:
                                mname_str += i + ' '
                        mname_str.replace(' ',',')
                        mname_str = mname_str[:-1]
                        mamount = 0
                        for i in range(3) :
                            if product[i] in purchase[track.track_id]:
                                mamount += product_price[i]
                            if product2[i] in purchase2[track.track_id]:
                                mamount += product2_price[i]
                    
                        webbrowser.open(url + '?mid=' + mid_str + '&mname=' + mname_str + '&mamount=' + str(mamount), 1)
                        payflag[track.track_id] = 1
                
                if x > xOver and x < width:
                    croppedImage=frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                    if len(croppedImage) == 0 : continue
                    inputHsv = cv2.cvtColor(croppedImage,cv2.COLOR_RGB2HSV) # RGB2HSV ?
                    # hist = cv2.calcHist([inputHsv],[0,1],None,[256],[0,256])
                    hist = cv2.calcHist([inputHsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    minHisOut = 1
                    matchedId = -1
                    for track2 in tracker.tracks:
                        bbox2 = track2.to_tlbr()
                        x2=int((bbox2[0]+bbox2[2])/2)
                        y2=int((bbox2[1]+bbox2[3])/2)
                        if x2 > width*0.9 and x2 < xOver + overWidth :
                            cropped=frame[int(bbox2[1]):int(bbox2[3]),int(bbox2[0]):int(bbox2[2])]
                            if len(cropped) == 0 : continue
                            cropHsv = cv2.cvtColor(cropped,cv2.COLOR_RGB2HSV)
                            hist2 = cv2.calcHist([cropHsv],[0,1],None,[180, 256], [0, 180, 0, 256])
                            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                            hisOut = cv2.compareHist(hist,hist2,cv2.HISTCMP_BHATTACHARYYA)
                            if minHisOut > hisOut:  
                                minHisOut = hisOut
                                matchedId = track2.track_id
                  
                    if matchedId is not -1 :
                        nametoTrackId[matchedId] = nametoTrackId[track.track_id]
                        purchase[track.track_id] = purchase[matchedId]
                        purchase2[track.track_id] = purchase2[matchedId]
                        #### arduino
            bag1,bag2 = arduino2(arduino_) 
            if bag1 :
                for i in userBuy_list_1:
                    if i in bag1:
                        bag1.remove(i)

            if bag2 :
                for i in userBuy_list_2:
                    if i in bag2:
                        bag2.remove(i)
            # dist_1 = [math.inf for i in range(50)] 
            # dist_2 = [math.inf for i in range(50)] 
            # for track in tracker.tracks :
            #     bbox = track.to_tlbr()
            #     x=int((bbox[0]+bbox[2])/2)
            #     y=int((bbox[1]+bbox[3])/2)
            #     dist_1[track.track_id] = (x-xCal)**2 + (y-yCal)**2
            #     dist_2[track.track_id] = (x-x2Cal)**2 + (y-y2Cal)**2
            # print(dist_1.index(min(dist_1))) 
            if bag1 :
                purchase[dist_1.index(min(dist_1))] =  [] + bag1
                if userBuy_1 != dist_1.index(min(dist_1)) :
                    for i in  purchase[dist_1.index(min(dist_1))]:
                        if i not in userBuy_list_1:
                            userBuy_list_1.append(i)
                    userBuy_1 = dist_1.index(min(dist_1))
            else:
                bag1=['none']
                purchase[dist_1.index(min(dist_1))] =  [] + bag1
                if userBuy_1 != dist_1.index(min(dist_1)) :
                    for i in  purchase[dist_1.index(min(dist_1))]:
                        if i not in userBuy_list_1:
                            userBuy_list_1.append(i)
                userBuy_1 = dist_1.index(min(dist_1))
            if bag2 :
                purchase2[dist_2.index(min(dist_2))] = [] + bag2
                
                if userBuy_2 != dist_2.index(min(dist_2)) :
                    for i in purchase2[dist_2.index(min(dist_2))]:
                        if i not in userBuy_list_2:
                            userBuy_list_2.append(i)
                    userBuy_2 = dist_2.index(min(dist_2))
            else :
                bag2=['none']
                purchase2[dist_2.index(min(dist_2))] = [] + bag2
                
                if userBuy_2 != dist_2.index(min(dist_2)) :
                    for i in purchase2[dist_2.index(min(dist_2))]:
                        if i not in userBuy_list_2:
                            userBuy_list_2.append(i)
                    userBuy_2 = dist_2.index(min(dist_2))
            # update tracks
            for track in tracker.tracks :
                if not track.is_confirmed() or track.time_since_update > 3:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                x=int((bbox[0]+bbox[2])/2)
                y=int((bbox[1]+bbox[3])/2)
                ####바운딩박스, text 등등 삽입####
                if str(type(nametoTrackId[track.track_id])) == "<class 'int'>" :
                    color = colors[(nametoTrackId[track.track_id]) % len(colors)]
                else :
                    color = colors[ord((nametoTrackId[track.track_id][0])) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2) ##유저bbox
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+120, int(bbox[1])), color, -1) #x,y좌표 box
                cv2.putText(frame,"x: "+str(x)+" y:"+str(y),(int(bbox[0]), int(bbox[1]-5)),0,0.5,(255,255,255),2) #x,y좌표
                cv2.rectangle(frame, (int(bbox[0]), y-20), (int(bbox[0])+len(str(nametoTrackId[track.track_id]))*13 , y+20), color, -1) #이름렉탱글
                cv2.putText(frame, str(nametoTrackId[track.track_id]),(int(bbox[0]), int(y)),0, 0.75, (255,255,255),2) #username
                cv2.putText(frame, str(purchase[track.track_id])+str(purchase2[track.track_id]),(int(bbox[0]), int(y+15)),0, 0.75, (255,255,255),2) #username  
                cv2.rectangle(frame, (int(bbox[0]), y-20), (int(bbox[0])+len(str())*13 , y+20), color, -1)
            ####FPS####
            fps = 1.0 / (time.time() - start_time) *2
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            ####이미지합치기####
            cv2.putText(result, str(int(fps)), (int(width), 50), 0,1, (255,255,255),2)

            ####네이밍장소,계산대그리기####
            cv2.rectangle(result, (int(xName), int(yName)), (int(xName+nameWidth), int(yName+nameHeight)), (0,0,255), 2)
            cv2.rectangle(result, (int(xCal), int(yCal)), (int(xCal+calWidth), int(yCal+calHeight)), (0,255,0), 2)
            cv2.rectangle(result, (int(x2Cal), int(y2Cal)), (int(x2Cal+calWidth2), int(y2Cal+calHeight2)), (0,255,0), 2)
            cv2.rectangle(result, (int(xQR), int(yQR)), (int(xQR+qrWidth), int(yQR+qrHeight)), (0,255,0), 2) # add

            cv2.imshow("Output Video", result)
            ####output파일저장####
            if FLAGS.output:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q') : break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)

    except SystemExit:
        pass