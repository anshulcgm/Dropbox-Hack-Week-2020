# Dropbox-Hack-Week-2020


The purpose of this project was to create a system such that the user could control the movement of a Tello EDU drone with hand movements and gestures. 

The code for the gesture recognition was adapted from Adrian Rosebrock's Raspberry Pi for Computer Vision Book (Highly Recommend). The Gesture Recognition code involves training a
Convolutional Neural Net on several predefined gestures such as the stop sign. To obtain the training data, I obtained samples from the webcam and cleaned the data using KNN Background Subtraction.

The most difficult part of the project was the Hand Tracking. To implement this I used frame subtraction along with contour detection and object tracking using OpenCV's built in object tracker.
After detecting the contour, I'd track it for 30 frames, figure out the direction it moved over those 30 frames and use that interface with the Tello EDU drone and move it in the appropriate direction.

One limitation of the project is that interfacing with the Tello EDU drone is rather unreliable and sometimes it moved in the wrong direction as a result. 
