# import the necessary packages
from PyInquirer import style_from_dict, Token, prompt
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import multiprocessing
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

if os.path.isdir("output") is False: os.mkdir("output")

def start_tracker(box, label, rgb, inputQueue, outputQueue):
    
    # construct a dlib rectangle object from the bounding box
    # coordinates and then start the correlation tracker
    t = dlib.correlation_tracker()
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    t.start_track(rgb, rect)
    
    # loop indefinitely -- this function will be called as a daemon
    # process so we don't need to worry about joining it
    while True:
        # attempt to grab the next frame from the input queue
        rgb = inputQueue.get()
        # if there was an entry in our queue, process it
        if rgb is not None:
            # update the tracker and grab the position of the tracked
            # object
            t.update(rgb)
            pos = t.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # add the label + bounding box coordinates to the output
            # queue
            outputQueue.put((label, (startX, startY, endX, endY)))

proto = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
model = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
    help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=50,
    help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

style = style_from_dict({ Token.QuestionMark: '#E91E63 bold', Token.Selected: '#00FFFF', Token.Instruction: '', Token.Answer: '#2196f3 bold', Token.Question: '#7FFF00 bold',})
time.sleep(0.2)
class_option=[ 
    {
        'type':'list',
        'name':'class',
        'message':'Class for tracking:',
        'choices': CLASSES,
    }
]
class_answer=prompt(class_option,style=style)
class_to_track=class_answer['class']

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(proto, model)

# if a video path was not supplied, grab a reference to the picamera
if not args.get("input", False):
    print("[INFO] starting video stream ...")
    vs = VideoStream(usePiCamera=True, resolution=(1080, 800)).start()
    time.sleep(2.0)
    
# otherwize, grab the inout video file
else :
    print("[INFO] opening video file ...")
    vs = cv2.VideoCapture(args["input"])
    
# initialize the video writer
writer = None

# set the name of the output video file corresponding to the input
if "/" in list(args["input"]):
    output = args["input"].split(".")[0].split("/")[-1]
else:
    output = args["input"].split(".")[0]

# initialize the frame dimension
W = None
H = None

# initialize our lists of queues -- both input queue and output queue
# for *every* object that we will be tracking
inputQueues = []
outputQueues = []

# initialize the centroid tracker
#ct = CentroidTracker(maxDisappeard = 40)
#trackableObjects = {}

# initialize the number of frames processed thus far
totalFrames = 0

# start the FPS estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # grab the next frame from the video file
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame
    
    #if we are viewing a video and we did not grab a frame we have to
    #reach the end of the video file
    if args["input"] is not None and frame is None:
        break
    
    # resize the frame for faster processing and then convert the
    # frame from BGR to RGB ordering (dlib needs RGB ordering)
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #if the frame dimension are empty then set them
    if W is None or H is None:
        (H,W) = frame.shape[:2]
    
    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output/"+output+".avi", fourcc, 30,
            (W, H), True)
        
    #initialize the status and the list of box
    status = "Waiting"
    #rects = []
    
    # check if we have to run the object detection method 
    if len(inputQueues) == 0 or totalFrames % args["skip_frames"] == 0:
        #set the status and pur new set of object trackers :
        status = "Detecting"
        inputQueues = []
        outputQueues = []
        
        # grab the frame dimensions and convert the frame to a blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (W, H), 127.5)

        # pass the blob through the network and obtain the detections
        # and predictions
        net.setInput(blob)
        detections = net.forward()
        
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                
                # if the class label is not a person, ignore it
                if CLASSES[idx] != class_to_track:
                    continue
                
                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                bb = (startX, startY, endX, endY)
                
                # create two brand new input and output queues,
                # respectively
                iq = multiprocessing.Queue()
                oq = multiprocessing.Queue()
                inputQueues.append(iq)
                outputQueues.append(oq)
                
                # spawn a daemon process for a new object tracker
                p = multiprocessing.Process(
                    target=start_tracker,
                    args=(bb, label, rgb, iq, oq))
                p.daemon = True
                p.start()
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                
                # grab the corresponding class label for the detection
                # and draw the bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
    # otherwise, we've already performed detection so let's track
    # multiple objects
    else:
        # loop over each of our input ques and add the input RGB
        # frame to it, enabling us to update each of the respective
        # object trackers running in separate processes
        for iq in inputQueues:
            iq.put(rgb)
        # loop over each of the output queues
        for oq in outputQueues:
            # set the status to tracking mode
            status = "Tracking"
            
            # grab the updated bounding box coordinates for the
            # object -- the .get method is a blocking operation so
            # this will pause our execution until the respective
            # process finishes the tracking update
            (label, (startX, startY, endX, endY)) = oq.get()
            
            # draw the bounding box from the correlation object
            # tracker
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
    #use the centroid tracker to associate the old object centroid with the newly
    #computed object centroids
    #objects = ct.update(rects)
    
    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)
        
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    # update the FPS counter and inscrease the number of frames
    totalFrames += 1
    fps.update()
    
    
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()
    
# if we are ,ot using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()
    
#otherwize release the vidoe file pointer
else:
    vs.release()
    
# do a bit of cleanup
cv2.destroyAllWindows()
     
            