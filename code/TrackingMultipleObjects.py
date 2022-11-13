from __future__ import print_function
import sys
import cv2
from random import randint
import time
import cv2
import timeit

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
font =  cv2.FONT_HERSHEY_PLAIN

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
    return tracker

videoPath = "../data/mot.mp4"
cap = cv2.VideoCapture(videoPath)

success, frame = cap.read()
if not success:
    sys.exit(1)

## Select boxes
bboxes = []
colors = []

success, image = cap.read()
if success:
    first_frame = image  # save frame as JPEG file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

bboxes = body_cascade.detectMultiScale(image=first_frame, scaleFactor=1.01, minNeighbors=10, minSize=(30, 30), maxSize=(2000,2000))
RGB=[(0,0,255),(0,140,255),(0,255,255),(0,128,0),(255,0,0),(130,0,75),(128,0,128)]

prevXY=[]
curXY=[]
for i in range(0, len(bboxes)):
    i = i%6
    colors.append(RGB[i])
    prevXY.append([int((bboxes[i][0]+bboxes[i][2]/2)),int((bboxes[i][1]+bboxes[i][3]/2))])
    curXY.append([int((bboxes[i][0]+bboxes[i][2]/2)),int((bboxes[i][1]+bboxes[i][3]/2))])

# Specify the tracker type
trackerType = "CSRT"
# Create MultiTracker object
multiTracker = cv2.legacy.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)

# Process video and track objects
outUser=[]
frameCnt=0
while cap.isOpened():
    frameCnt=frameCnt+1
    success, frame = cap.read()
    if not success:
        break

    startTime = timeit.default_timer()
    if(frameCnt%2==1):
        continue    
    success, boxes = multiTracker.update(frame)

    for user, newbox in enumerate(boxes): #user : user번호, newbox : 해당 유저의 bounding box [x,y,w,h]
        if user not in outUser :
            p1 = (int(newbox[0]), int(newbox[1])) #시작점 좌표
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3])) #끝점 좌표
            curXY[user][0] = int((p1[0]+p2[0])/2)
            curXY[user][1] = int((p1[1]+p2[1])/2)
            #print('%d번의 좌표 %d,%d' %(user, curXY[0],curXY[1]))
            if (curXY[user][0]-int(prevXY[user][0]) <= 30) and (curXY[user][1]-int(prevXY[user][1]) <= 30) : 
                    prevXY[user][0] = curXY[user][0]
                    prevXY[user][1] = curXY[user][1]
                    cv2.rectangle(frame, p1, p2, colors[user], 2, 1)
                    text=str(user)
                    frame = cv2.putText(frame, text, (p2[0], p2[1]), font, 2, RGB[user],2)
            else :
                    print("%d user out indexing" %user)
                    outUser.append(user)
                    break

    #frame check
    termTime = timeit.default_timer()
    FPS = str(int(1./(termTime - startTime )))
    frame = cv2.putText(frame, FPS, (width-150, 100), font, 8, RGB[0],3)

    cv2.imshow('MultiTracker', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break