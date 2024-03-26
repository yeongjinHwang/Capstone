## Overview
해당 프로젝트는 yolo기반으로 MTMC 라는 기술을 개발한 프로젝트입니다.(대학교 졸업 캡스턴)
1. 두 camera의 image를 받아오고 multi tracking, 각 user bbox를 image로 잘라 색상정보 저장
2. 다른 camera에 user가 포착되면 이전 camera에 포착된 user data기반으로 색상정보가 비슷한 user는 <br\>
   같은 user라고 판단하여 MTMC
3. 추가적으로 이를 AI 편의점에 도입할 수 있을 것이라고 판단, 계산대에 무게 변화를 event로 감지하고
4. event가 생기면 그 계산대와 가까운 user 정보에 무게기반으로 물품명과 개수를 등록
5. 편의점에서 나간다고 판단되는 (출구 camera에 캡쳐된) user는 user정보에 등록되어 있는 물품기반으로 <br\>
   가격을 책정하고 kakaotalk을 통해 QR코드 전송(이 기능은 local pc에 QR정보까지만 구현)
6. QR정보에는 물품명, 물품개수의 형태로 나열

-> 해당 프로젝트 기반 한국정보기술학회 대학생 논문 개제

## Getting Started
$ git clone https://github.com/theAIGuysCode/yolov4-deepsort.git  

$ cd yolov4-deepsort  

## Conda 
conda-gpu.yml에서 tensorflow-gpu==2.3.0rc0 -> tensorflow-gpu==2.3.0

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

#### webcam에서 tracker 수행
$ python object_tracker.py --video1 $videoNum1 --video2 $videoNum2 --output ./outputs/webcam.avi --model yolov4   

or   

$ python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video1 0 --video2 2 --output ./outputs/tiny.avi --tiny   

## Frame
yolo cpu6 gpu13   

yoloTiny cpu14 gpu30   

