# python Multi_Object_Detection_Tracking_Centroid -i input/
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
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

if os.path.isdir("output") is False: os.mkdir("output")

labels = 'mobilenet_ssd/coco_labels.txt'
model = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
    help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
ap.add_argument("-k", "--top_k", type=int, default=3,
    help='number of categories with highest score to display')
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
    help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
#CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#    "sofa", "train", "tvmonitor"]

#style = style_from_dict({ Token.QuestionMark: '#E91E63 bold', Token.Selected: '#00FFFF', Token.Instruction: '', Token.Answer: '#2196f3 bold', Token.Question: '#7FFF00 bold',})
#time.sleep(0.2)
#class_option=[ 
#    {
#        'type':'list',
#        'name':'class',
#        'message':'Class for tracking:',
#        'choices': CLASSES,
#    }
#]
#class_answer=prompt(class_option,style=style)
#class_to_track=class_answer['class']

# load our serialized model from disk
print("[INFO] loading model...")
interpreter = make_iterpreter(model)
interpreter.allocate_tensors()
labels = read_label_file(labels)
inference_size = input_size(interpreter)

# if a video path was not supplied, grab a reference to the picamera
if not args.get("input", False):
    print("[INFO] starting video stream ...")
    vs = VideoStream(usePiCamera=True, resolution=(640, 480)).start()
    time.sleep(2.0)
    output = "output"
    
# otherwize, grab the input video file
else :
    print("[INFO] opening video file ...")
    vs = cv2.VideoCapture(args["input"])
    if "/" in list(args["input"]):
        output = args["input"].split(".")[0].split("/")[-1]
    else:
        output = args["input"].split(".")[0]
    
# initialize the video writer
writer = None

# initialize the frame dimension
W = None
H = None


# initialize the centroid tracker
ct = CentroidTracker()
trackers = []
trackableObjects = {}

# initialize the number of frames processed thus far
totalFrames = 0

# start the FPS estimator
fps = FPS().start()


# loop over frames from the video file stream
while True:
    try:
        # grab the next frame from the video file
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        
        #if we are viewing a video and we did not grab a frame we have to
        #reach the end of the video file
        if args["input"] is not None and frame is None:
            break
        
        # resize the frame for faster processing and then convert the
        # frame from BGR to RGB ordering (dlib needs RGB ordering)
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = imutils.resize(cv2_im_rgb, inference_size)
        
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
        rects = []
        
        # check if we have to run the object detection method 
        if totalFrames % args["skip_frames"] == 0:
            #set the status and pur new set of object trackers :
            status = "Detecting"
            trackers = []
            
            # grab the frame dimensions and convert the frame to a blob
            #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (W, H), 127.5)

            # pass the blob through the network and obtain the detections
            # and predictions
            #net.setInput(blob)
            #detections = net.forward()
            detections = get_objects(interpreter, args.confidence)[:args.top_k]
            
            height, width, channels = frame.shape
            scale_x, scale_y = width / inference_size[0], height / inference_size[1]
            
            # loop over the detections
            for obj in detections:
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = int(100 * obj.score)
                label = labels.get(obj.id, obj.id)
                # filter out weak detections by requiring a minimum
                # confidence
                #if confidence > args["confidence"]:
                    # extract the index of the class label from the
                    # detections list
                    #idx = int(detections[0, 0, i, 1])
                    #label = CLASSES[idx]
                    
                    # if the class label is not a person, ignore it
                    #if label != class_to_track:
                        #continue
                    bbox = obj.bbox.scale(scale_x, scale_y)
                    x0, y0 = int(bbox.xmin), int(bbox.ymin)
                    x1, y1 = int(bbox.xmax), int(bbox.ymax)
                    
                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    #box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    #(startX, startY, endX, endY) = box.astype("int")
                    bb = (x0, y0, x1, y1)
                    
                    # create two brand new input and output queues,
                    # respectively
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
                    tracker.start_track(rgb, rect)
                    
                    #add the tracker to the list
                    trackers.append(tracker)
                    
                
        # otherwise, we've already performed detection so let's track
        # multiple objects
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status to tracking mode
                status = "Tracking"
                
                #update the tracker
                tracker.update(rgb)
                pos = tracker.get_position()
                
                #unpack the position
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                
                #add the bounding box coordinates to the recatngle list
                rects.append((startX, startY, endX, endY))
                
        #use the centroid tracker to associate the old object with the new object
        objects = ct.update(rects)
                
        #loop over the tracked object
        for (objectID, centroid) in objects.items():
            #check to see if a trackable object exist for the current id
            to = trackableObjects.get(objectID, None)
                    
            #if there is no tracable object create one
            if to is None :
                to = TrackableObject(objectID, centroid)
                        
            #otherwize there is a trackable object that we can utilize
            else:
                to.centroids.append(centroid)
                    
            #store the trackable object in our ditionnary
            trackableObjects[objectID] = to
                    
            # draw the bounding box from the correlation object
            # tracker
            label = "ID {}".format(objectID)
            cv2.putText(frame, label, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                    
        #construct info for display
        text = "{}".format(status)
        cv2.putText(frame, text, (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
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
        
    except KeyboardInterrupt :
        break
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
