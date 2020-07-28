from pyimagesearch.utils import Conf
from imutils.video import VideoStream
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime
from datetime import date
from time import sleep
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import tellopy

def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--conf", required = True, help = "path to input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

print("[INFO] loading model...")
model = load_model(str(conf["model_path"]))
lb = pickle.loads(open(str(conf["lb_path"]), "rb").read())

print("[INFO] starting video stream thread...")

vs = VideoStream(src = 0).start()
time.sleep(2.0)

currentGesture = [None, 0]

gestures = []

drone = tellopy.Tello()
drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
drone.connect()
drone.wait_for_connection(60.0)


canTakeoff = True
canLand = False

canMoveRight = False
canMoveLeft = False
firstFrame = None
while True:
    frame = vs.read()


    timestamp = datetime.now()
    frame = imutils.resize(frame, width = 500)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gray
        continue
    clone = frame.copy()
    frameDelta = cv2.absdiff(firstFrame, gray)
    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

    roi = frameDelta[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.threshold(roi, 25, 255, cv2.THRESH_BINARY)[1]
    visROI = roi.copy()

    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis = 0)

    proba = model.predict(roi)[0]
    label = lb.classes_[proba.argmax()]
    print(label)
    if label == "fist" and canTakeoff:
        print("Trying to takeoff")
        drone.takeoff()
        sleep(5)
        canTakeoff = False
        canLand = True
        canMoveLeft = True
        canMoveRight = True
    elif label == "ignore" and canLand:
        print("Trying to land")
        drone.land()
        #sleep(5)
        canLand = False
        canMoveLeft = False
        canMoveRight = False
        canTakeoff = True
    elif label == "left" and canMoveLeft:
        print("Moving left")
        drone.left(20)
        sleep(5)
        canMoveLeft = False
        canMoveRight = True
    elif label == "right" and canMoveRight:
        print("Moving right")
        drone.right(20)
        sleep(5)
        canMoveRight = False
        canMoveLeft = True
    cv2.imshow("Frame", clone)
    cv2.imshow("ROI", visROI)
    cv2.imshow("Frame difference", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

