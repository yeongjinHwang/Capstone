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

####GPU로 쓸게요####
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video1
    video_path2 = FLAGS.video2

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    ####비디오 캡쳐####
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
    namingX, namingY= xName+nameWidth/2, yName+nameHeight/2
    medianX, medianY, medianX2, medianY2 = xCal+calWidth/2, yCal+calHeight/2, x2Cal+calWidth2/2, y2Cal+calHeight2/2
    print("naming location : x "+str(xName)+" y "+str(yName)+" width "+str(nameWidth)+" height "+str(nameHeight)+\
    " namingX : "+str(namingX)+' namingY : '+str(namingY))
    print("cal1 location : x "+str(xCal)+" y "+str(yCal)+" width "+str(calWidth)+" height "+str(calHeight)+\
    " medianX : "+str(medianX)+' medianY : '+str(medianY))
    print("cal2 location : x "+str(x2Cal)+" y "+str(y2Cal)+" width "+str(calWidth2)+" height "+str(calHeight2)+\
    " medianX : "+str(medianX2)+' medianY : '+str(medianY2))

    ####--output path 인자로 시작하면 저장하기 위한 코드####
    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    ####영상or웹캠실행####
    frameDrop=0
    nameBuf=[]
    while True:
        frameDrop=frameDrop+1
        while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            name = sys.stdin.readline()
            if name:
                nameBuf.append(name[:-1])
        if frameDrop%2==0 :
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
            count = len(names)

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
            # update tracks
            for track in tracker.tracks :
                if not track.is_confirmed() or track.time_since_update > 3:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                x=int((bbox[0]+bbox[2])/2)
                y=int((bbox[1]+bbox[3])/2)
                ####바운딩박스, text 등등 삽입####
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame,"x: "+str(x)+" y:"+str(y),(int(bbox[0]), int(bbox[1]-5)),0,0.5,(255,255,255),2)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-20)),0, 0.75, (255,255,255),2)
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

            imgPath = './black.jpg'
            blackImg = cv2.resize(cv2.imread(imgPath),(int(width*2),200))
            result = cv2.vconcat([result, blackImg])
            #### 유저 구매관리 ####
            if len(nameBuf)>0:
                for i in range(len(nameBuf)) :
                    cv2.putText(result, nameBuf[i]+':', (40+(i//6)*400, int(height)+30+(i%6)*30), 0,0.75, (255,255,255),2)

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