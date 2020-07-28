from configFiles.utils import Conf
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import numpy as np
from src import utils
from src.model_color_mask import ColorMask
from src.model_opencv_subtractor import OpenCVSubtractor
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "path to the input configuration file")
ap.add_argument("-b", "--background-subtractor", required = True, help = "Background Subtraction method to use")
args = vars(ap.parse_args())


conf = Conf(args["conf"])

TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

MAPPINGS = conf["mappings"]

for (key, label) in list(MAPPINGS.items()):

	MAPPINGS[ord(key)] = label
	del MAPPINGS[key]

validKeys = set(MAPPINGS.keys())

if args["background_subtractor"] == "Gaussian":
    bg_sub = cv2.createBackgroundSubtractorMOG2()
elif args["background_subtractor"] == "KNN":
    bg_sub = cv2.createBackgroundSubtractorKNN()
else:
    print("Your input was invalid")

keyCounter = {}

print("[INFO] starting video stream thread...")
vs = VideoStream(src = 0).start()

firstFrame = None
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = cv2.flip(frame, 1)
	img_mask = bg_sub.apply(frame)
	img_blur = cv2.GaussianBlur(img_mask, (15, 15), 0)
	_, img_res = cv2.threshold(img_blur, 127, 255, cv2.THRESH_OTSU)
	h, w = img_mask.shape
	shape = (h, 2 * w)
	img_out = np.zeros(shape, np.uint8)
	img_out[:, :w] = img_mask
	img_out[:, w:] = img_res
	cv2.imshow("Result", img_out)
	roi = img_mask[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
	cv2.imshow("ROI", roi)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	elif key in validKeys:
		p = os.path.sep.join([conf["dataset_path"], MAPPINGS[key]])

		if not os.path.exists(p):
			os.mkdir(p)
	
		p = os.path.sep.join([p, "{}.png".format(keyCounter.get(key, 0))])
		keyCounter[key] = keyCounter.get(key, 0) + 1

		print("[INFO] saving ROI: {}".format(p))
		cv2.imwrite(p, roi)
"""
while True:
	frame = vs.read()

	frame = imutils.resize(frame, width = 500)
	frame = cv2.flip(frame, 1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img_mask = bg_sub.apply(gray)
	img_blur = cv2.GaussianBlur(img_mask, (15, 15), 0)
	_, img_res = cv2.threshold(img_blur, 127, 255, cv2.THRESH_OTSU)
	h, w = img_mask.shape
	shape = (h, 2 * w)
	img_out = np.zeros(shape, np.uint8)
	img_out[:, :w] = img_mask
	img_out[:, w:] = img_res
	#roi = gray[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
	#roi = cv2.threshold(roi, 127, 255, cv2.THRESH_OTSU)[1]
	
	#cv2.imshow("Result", result)
	cv2.imshow("Frame", frame)
	#cv2.imshow("ROI", roi)
	cv2.imshow("Image output", img_out)
	#if firstFrame is None:
		#firstFrame = gray
		#continue
	#clone = frame.copy()
	#frameDelta = cv2.absdiff(firstFrame, gray)
	#cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)
	#roi = frameDelta[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
	#roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	#roi = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY)[1]

	
	
	#cv2.imshow("Frame", clone)
	#cv2.imshow("ROI", roi)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	elif key in validKeys:
		p = os.path.sep.join([conf["dataset_path"], MAPPINGS[key]])

		if not os.path.exists(p):
			os.mkdir(p)
		
		p = os.path.sep.join([p, "{}.png".format(keyCounter.get(key, 0))])
		keyCounter[key] = keyCounter.get(key, 0) + 1

		print("[INFO] saving ROI: {}".format(p))
		#cv2.imwrite(p, roi)
"""