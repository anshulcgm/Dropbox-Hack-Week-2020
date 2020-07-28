from configFiles.utils import Conf
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np



def applyFrameTransformation(frame, bg_sub, TOP_LEFT, BOT_RIGHT):
    flippedFrame = cv2.flip(frame, 1)
    img_mask = bg_sub.apply(flippedFrame)
    print(f"Mask shape is {img_mask.shape}")
    roi = img_mask[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
    print(f"ROI shape is {roi.shape}")
    roi = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2GRAY)
    return roi
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "path to input configuration file")
ap.add_argument("-b", "--background-subtractor", required = True, help = "Background Subtractor: Gaussian or KNN")
args = vars(ap.parse_args())

feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


conf = Conf(args["conf"])

TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

if args["background_subtractor"] == "Gaussian":
    bg_sub = cv2.createBackgroundSubtractorMOG2()
elif args["background_subtractor"] == "KNN":
    bg_sub = cv2.createBackgroundSubtractorKNN()
else:
    print("Background Subtractor input was invalid")

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_roi = applyFrameTransformation(prev_frame, bg_sub, TOP_LEFT, BOT_RIGHT)
prev_features = cv2.goodFeaturesToTrack(prev_roi, mask = None, **feature_params).astype("float32")
mask = np.zeros_like(prev_roi)

color = (0, 255, 0)

while (cap.isOpened()):
    ret, frame = cap.read()

    roi = applyFrameTransformation(frame, bg_sub, TOP_LEFT, BOT_RIGHT)

    next, status, error = cv2.calcOpticalFlowPyrLK(prev_roi, roi, prev_features, None, **lk_params)

    good_old = prev_features[status == 1]
    good_new = next[status == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c,d = old.ravel()

        mask = cv2.line(mask, (a, b), (c, d), color, 2)
        frame = cv2.circle(roi, (a, b), 3, color, -1)
    
    output = cv2.add(roi, mask)

    prev_roi = roi.copy()

    prev_features = good_new.reshape(-1, 1, 2)
    cv2.imshow("Sparse Optical Flow", output)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()