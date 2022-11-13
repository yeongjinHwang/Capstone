## Getting Started
$ git clone https://github.com/theAIGuysCode/yolov4-deepsort.git  

$ cd yolov4-deepsort  

## Conda 
conda-gpu.yml에서 tensorflow-gpu==2.3.0rc0 -> tensorflow-gpu==2.3.0rc0   

$ conda env create -f conda-gpu.yml  

$ conda activate yolov4-gpu  

## Pre-trained Weights
80개 이상의 클래스에서 yolo v4에  대해 이미 훈련된 모델로 80개 이상의 클래스를 감지할 수 있음  

yolov4      : https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights  

yolov4 tiny : https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights  

yolov4-deepsort/data/ 에 저장   

## Run
#### 다크넷 웨이트를 tensorflow model로 변환
$ python save_model.py --model yolov4  

or  

$ python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny  

#### video에서 tracker 수행
$ python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4  

or   

$ python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny   

#### webcam에서 tracker 수행
$ python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4   

or   

$ python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video 0 --output ./outputs/tiny.avi --tiny   

## Frame
yolo cpu6 gpu13   

yoloTiny cpu14 gpu30   

