from configFiles.utils import Conf
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2 
import os
import numpy as np
import tellopy
from time import sleep
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pickle

def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "config file")
ap.add_argument("-v", "--video", help = "path to the video file")
ap.add_argument("-a", "--area", default = 10000, help = "Minimum area size")
ap.add_argument("-s", "--skip", default = 10000000, help = "number of frames to skip in between contour detection")
args = vars(ap.parse_args())
conf = Conf(args["conf"])

TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

model = load_model(str(conf["model_path"]))
lb = pickle.loads(open(str(conf["lb_path"]), "rb").read())

lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = (0, 255, 0)

tracker = cv2.TrackerCSRT_create()
firstFrame = None
prevROI = None
firstContour = None
prevBoxLoc = None
counter = 1

#drone = tellopy.Tello()
#drone.subscribe(drone.EVENT_FLIGHT_DATA, handler) 
#drone.connect()
#drone.wait_for_connection(60.0)

canTakeoff = True
canLand = False

canMoveRight = False
canMoveLeft = False

canMoveForward = False
canMoveBackward = False
action = "Waiting for input"
if args.get("video", None) is None:
    vs = VideoStream(src = 0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])
mask = None
startTime = None
firstTrackedLocation = None
fps = FPS().start()
mode = "motion"
while True:
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

    if frame is None:
        break
    frame = cv2.flip(frame, 1) 
    #print(f"Frame shape is {frame.shape}")
    frame = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
    frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue
    if startTime is None:
        startTime = time.time()
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    if mode == "motion":
        thresh = cv2.dilate(thresh, None, iterations = 2)
        if firstContour is None:
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts) != 0:
                largestCnt = max(cnts, key = cv2.contourArea)

                if cv2.contourArea(largestCnt) >= args["area"]:
                    print(f"Area of largest contour is {cv2.contourArea(largestCnt)}")
                    firstContour = largestCnt
                    (x, y, w, h) = cv2.boundingRect(largestCnt)
                    #hull = cv2.convexHull(largestCnt)
                    """
                    if prevBoxLoc is None:
                        prevBoxLoc = np.array([x, y])
                    else:
                        print(f"Previous box location was {prevBoxLoc} and current box location is ({x}, {y})")
                        currentBoxLoc = np.array([x, y])
                        displacement = currentBoxLoc - prevBoxLoc
                        print(f"Displacement is {displacement}")
                        if abs(displacement[0]) > abs(displacement[1]):
                            if displacement[0] < -frame.shape[1] / 5:
                                action = "Moving Left"
                                if canMoveLeft:
                                    #drone.left(10)
                                    #sleep(2)
                                    canMoveLeft = False
                                    canMoveRight = True
                            elif displacement[0] > frame.shape[1] / 5:
                                action = "Moving Right"
                                if canMoveRight:
                                    #drone.right(10)
                                    #sleep(2)
                                    canMoveRight = False
                                    canMoveLeft = True
                            else:
                                action = "Waiting for input - can't move left or right"
                        elif abs(displacement[1]) > abs(displacement[0]):
                            if displacement[1] < -frame.shape[0] / 5:
                                action = "Moving Up"
                                if canTakeoff:
                                    #drone.takeoff()
                                    #sleep(2)
                                    canTakeoff = False
                                    canLand = True
                                    canMoveLeft = True
                                    canMoveRight = True
                                    canMoveForward = True
                                    canMoveBackward = True
                            elif displacement[1] > frame.shape[0] / 5:
                                action = "Moving down"
                                if canLand:
                                    #drone.land()
                                    canLand = False
                                    canMoveLeft = False
                                    canMoveRight = False
                                    canMoveBackward = False
                                    canMoveForward = False
                                    canTakeoff = True
                            else:
                                action = "Waiting for input - can't move up or down"
                        else:
                            action = "Waiting for input - no significant displacement"
                        print(f"Action is {action}")
                        prevBoxLoc = currentBoxLoc
                    track = tracker.init(frame, (x, y, w, h))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    """
                    track = tracker.init(frame, (x, y, w, h))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    #cv2.drawContours(frame, hull, 0, (0, 255, 0))
                    firstTrackedLocation = None
                    startTime = time.time()
            else:
                action = "Waiting for hand motion"
                

        else:
            if counter % args["skip"] == 0:
                """
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                if len(cnts) != 0:
                    largestCnt = max(cnts, key = cv2.contourArea)

                    if cv2.contourArea(largestCnt) >= args["area"]:
                        (x, y, w, h) = cv2.boundingRect(largestCnt)
                        track = tracker.init(frame, (x, y, w, h))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                """
                firstContour = None
                
            else:
                ok, bbox = tracker.update(frame)

                if ok:
                    #print("Tracking was a success")
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    (x, y, w, h) = bbox
                    if firstTrackedLocation is None:
                        firstTrackedLocation = np.array([x, y])
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

                    if counter % args["skip"] == (args["skip"] - 1):
                        currentLocation = np.array([x, y])
                        displacement = currentLocation - firstTrackedLocation
                        if abs(displacement[0]) > abs(displacement[1]):
                            if displacement[0] < -frame.shape[1] / 5:
                                action = "Moving Left"
                                if canMoveLeft:
                                    #drone.left(8)
                                    #sleep(2)
                                    canMoveLeft = False
                                    canMoveRight = True
                            elif displacement[0] > frame.shape[1] / 5:
                                action = "Moving Right"
                                if canMoveRight:
                                    #drone.right(8)
                                    #sleep(2)
                                    canMoveRight = False
                                    canMoveLeft = True
                            else:
                                action = "Waiting for input - can't move left or right"
                        elif abs(displacement[1]) > abs(displacement[0]):
                            if displacement[1] < -frame.shape[0] / 5:
                                action = "Moving Up"
                                if canTakeoff:
                                    #drone.takeoff()
                                    #sleep(2)
                                    canTakeoff = False
                                    canLand = True
                                    canMoveLeft = True
                                    canMoveRight = True
                                    canMoveForward = True
                                    canMoveBackward = True
                            elif displacement[1] > frame.shape[0] / 5:
                                action = "Moving down"
                                if canLand:
                                    #drone.land()
                                    canLand = False
                                    canMoveLeft = False
                                    canMoveRight = False
                                    canMoveBackward = False
                                    canMoveForward = False
                                    canTakeoff = True
                            else:
                                action = "Waiting for input - can't move up or down"
            
            counter += 1
    elif mode == "gesture":
        visualizeThresh = thresh.copy()
        thresh = cv2.resize(thresh, (64, 64))
        thresh = thresh.astype("float") / 255.0
        thresh = img_to_array(thresh)
        roi = np.expand_dims(thresh, axis = 0)

        proba = model.predict(roi)[0]
        label = lb.classes_[proba.argmax()]
        print(f"Detected label is {label}")
        if label == "fist":
            action = "Move forward"
            #drone.forward(5)
            canMoveForward = False
            canMoveBackward = True
        elif label == "stop":
            action = "Move backward"
            #drone.backward(5)
            canMoveBackward = False
            canMoveForward = True
        else:
            action = "Waiting for gestures"
    """
    for c in cnts:
        if cv2.contourArea(c) < args["area"]:
            continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """
    cv2.putText(frame, "Status: {}".format(action), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        #drone.land()
        break
    elif key == ord("g"):
        mode = "gesture"
    elif key == ord("m"):
        mode = "motion"
    fps.update()

fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
#drone.quit()