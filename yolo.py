import cv2
import numpy as np 
import requests
import json
import math
from datetime import datetime



def fromHourToSecond(curTime):
    listTimeString=curTime.split(":")
    listTimeInt=[]
    for i in listTimeString:
        listTimeInt.append(int(i))
        
    seconds=listTimeInt[0]*3600+listTimeInt[1]*60+listTimeInt[2]    
    return seconds 
    
def getCurTimeSecond():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    curSecond=fromHourToSecond(current_time)
    return curSecond

def getTime(prevTime,curTime): 
    myTime=abs(curTime-prevTime)
    return myTime


def getSpeed(prevPos,curPos,prevTimeSeconds,curTimeSeconds):
    
    x = curPos[0] - prevPos[0]  
    y = curPos[1] - prevPos[1] 
    
    distance = math.sqrt(abs(x * x - y * y))
    return round(distance / getTime(prevTimeSeconds,curTimeSeconds))
    






def setImage(url,path):
    receive = requests.get(url)
    with open(path,'wb') as f:
        f.write(receive.content)



#Load yolo
def load_yolo():
	net = cv2.dnn.readNet("C:/Users/ibrahim.l/Desktop/object-detection-yolo-opencv-master/yolov3.weights", "C:/Users/ibrahim.l/Desktop/object-detection-yolo-opencv-master/yolov3.cfg")
	classes = []
	with open("C:/Users/ibrahim.l/Desktop/object-detection-yolo-opencv-master/coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img,(1200,600))
	height, width, channels = img.shape
	return img, height, width, channels




def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids



			
def draw_labels(boxes, confs, colors, class_ids, classes, img):
    numPerson=0
    bigListTime=[]
    bigListPos=[]
    speed=0
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    data = {}
    for i in range(len(boxes)):
        if i in indexes:
            TimeTuple=[0,0]  
            PosTuple=[(0,0),(0,0)]
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            #count persons on the frame
            if(label=='person'):
                numPerson+=1
                
                
            data[label] = []
            color = colors[i]
            ##positon and time to calculate the speed
            interPos=PosTuple[1]
            PosTuple[0]=interPos
            PosTuple[1]=(x,y)
            
            interTime=TimeTuple[1]
            TimeTuple[0]=interTime
            TimeTuple[1]=getCurTimeSecond()
            
            #bigListTime.append(TimeTuple)
            #bigListPos.append(PosTuple)
            
            #speed=getSpeed(PosTuple[0],PosTuple[1],TimeTuple[0],TimeTuple[1])
            center = (round(x + (w / 2)), round(y + (h / 2)))
            data[label].append({
                    'Position': str(center),
                    
                    })
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.circle(img, center, 5, (0, 0, 255), 5)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
            cv2.putText(img,str("numPersons:"+str(numPerson)), (10,10), font, 1, color, 1)
            cv2.imshow("Image", img)
            
            with open('data.txt', 'w') as outfile:
                json.dump(data, outfile)
        print(data)
    cv2.imshow("Image", img)
        
            



def image_detect_cam(url,path):
    while True:
        setImage(url,path)
        model, classes, colors, output_layers = load_yolo()
        image, height, width, channels = load_image(path)
        blob, outputs = detect_objects(image, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, image)
        key = cv2.waitKey(1)
        if key == 27:
            break





def yoloWorker(parameterlist):

    #visualParameters
    visualizeBBoxes = parameterlist[0]
    visualizerCenters = parameterlist[1]

    #calculationParameters
    calculateDirection = parameterlist[2]
    calculateSpeed = parameterlist[3]
    calculatePeopleCount = parameterlist[4]
    calculateTotalPeopleCount = parameterlist[5]
    classList = parameterlist[6]
    calculateLineCrossed = parameterlist[7]
    videoSource = parameterlist[8]

    print(classList)

    #videopath = '/home/openremote/Desktop/pytorch_objectdetecttrack-master/video.mp4'

    #setup 
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

    if videoSource == "0" or videoSource == "1":
        vid = cv2.VideoCapture(int(videoSource))
    else:
        vid = cv2.VideoCapture(videoSource)

    mot_tracker = Sort() 

    pointsDict = {}
    TrackedIDs = []
    lineCrossingIDs = [] #list of ID's which are currantly crossing the line

    #parameters saved each frame
    prevPeopleCount = 0
    totalPeopleCount = 0

    totalLineCrossedLeft = 0
    totalLineCrossedRight = 0
    totalLineCrossed = 0

    frames = 0

    #START!
    while(True):

        peoplecount = 0
        
        ret, frame = vid.read()
        if not ret:
            break

        frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg, classList)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        

        #total parameters
        totalSpeed = 0
        left = 0
        right = 0
        up = 0
        down = 0
        

        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())
            #unique_labels = detections[:, -1].cpu().unique()

            #currentlyTrackedId = []

            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            
                speed = 0
                xdir = ""
                ydir = ""

                #get bounding box cordinates
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                #calculate center of object
                center = (round(x1 + (box_w / 2)), round(y1 + (box_h / 2)))
                
                #get ID
                Id = int(obj_id)

                #if Id not in currentlyTrackedId:
                #    currentlyTrackedId.append(Id)

                if calculatePeopleCount:
                    peoplecount += 1 
                    if Id not in TrackedIDs:
                        TrackedIDs.append(Id)

                #add center to dict
                if Id in pointsDict:
                    pointsDict[Id].appendleft(center)

                else:
                    #if(len(pointsDict) > maxPointsDictLength):
                    #    del list(pointsDict)[0]
                    #    print("delete")
                    pointsDict[Id] = deque(maxlen=25)
                    pointsDict[Id].appendleft(center)

                if len(pointsDict[Id]) > 6:
                    if calculateDirection:
                        xdir, ydir = getDirection(frame, pointsDict[Id])
                        
                        if(xdir == "left"):
                            left += 1
                        if(xdir == "right"):
                            right += 1
                        if(ydir == "up"):
                            up += 1
                        if(ydir == "down"):
                            down += 1


                    if calculateSpeed:
                        speed = getSpeed(pointsDict[Id])
                        totalSpeed += speed

                    if(frames % 10 == 0):
                        if calculateLineCrossed:
                            lineCrossed = getCountLineCrossed(pointsDict[Id])
                            if lineCrossed != None:
                                if Id not in lineCrossingIDs:
                                    if lineCrossed == "left":
                                        totalLineCrossedLeft += 1
                                    elif lineCrossed == "right":
                                        totalLineCrossedRight += 1
                                    totalLineCrossed += 1
                                    lineCrossingIDs.append(Id)
                            else:
                                if Id in lineCrossingIDs:
                                    lineCrossingIDs.remove(Id)

                #visualize boxes
                if visualizeBBoxes:
                    color = colors[Id % len(colors)]
                    cls = classes[int(cls_pred)]
                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                    cv2.rectangle(frame, (x1, y1-105), (x1+len(cls)*19+80, y1), color, -1)
                    cv2.putText(frame, cls + "-" + str(Id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                    
                    if calculateDirection:
                        cv2.putText(frame, xdir  + " - " + ydir, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

                    if calculateSpeed:
                        cv2.putText(frame, "speed " + str(speed), (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

                #visualize centers
                if visualizerCenters:
                    cv2.circle(frame, center, 5, (0, 0, 255), 5)
            
        # clean up unused Id's
        #if(frames % 90 == 0):
        #    idsToRemove = []
        #    for Id in pointsDict: 
        #        if Id not in currentlyTrackedId:
        #            idsToRemove.append(Id)
        #    for Id in idsToRemove:
        #        del pointsDict[Id]
        #visualize line
        if calculateLineCrossed:
            cv2.line(frame, (0,318), (637,221), [0, 255, 0], 10)
            cv2.putText(frame, "poeple count line crossed to left " + str(totalLineCrossedLeft), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)            
            cv2.putText(frame, "poeple count line crossed to right " + str(totalLineCrossedRight), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)            
            cv2.putText(frame, "poeple count line crossed Total " + str(totalLineCrossed), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)            
            
        # visualize People count
        if calculatePeopleCount:
            cv2.putText(frame, "people count " + str(peoplecount), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)            
       
        #get total people count
        if calculateTotalPeopleCount:
            #if peoplecount > prevPeopleCount:
            #    totalPeopleCount += abs(peoplecount - prevPeopleCount)
            #prevPeopleCount = peoplecount
            cv2.putText(frame, "total people count " + str(len(TrackedIDs)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)            

        #get total direction
        if calculateDirection:
            totalxdir, totalydir = getTotalDirection(left,right,up,down)
            cv2.putText(frame, "Total Xdir " + totalxdir, (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)            
            cv2.putText(frame, "total Ydir " + totalydir, (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)   


        #get Average speed
        if peoplecount != 0:
            totalSpeed/peoplecount
        else:
            totalSpeed = 0

        #write json for API
        if(frames % 10 == 0):
            writeJson(peoplecount, len(TrackedIDs), totalSpeed, totalxdir, totalydir, totalLineCrossedLeft, totalLineCrossedRight, totalLineCrossed)

        #visualize
        #frame = cv2.resize(frame, (1920,1080))
        cv2.imshow('Stream', frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break



    

image_detect_cam("http://192.168.1.100:8080/photo.jpg","C:/Users/ibrahim.l/Desktop/object-detection-yolo-opencv-master/images/myImage.jpg")
	

cv2.destroyAllWindows()
