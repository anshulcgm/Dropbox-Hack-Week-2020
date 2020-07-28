from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from configFiles.nn.gesturenet import GestureNet
from configFiles.utils import Conf
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("-c", "--conf", required = True, help = "path to input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

print("[INFO] loading images...")
imagePaths = list(paths.list_images(conf["dataset_path"]))
data = []
labels = []

for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (64, 64))

	data.append(image)
	labels.append(label)

data = np.array(data, dtype = "float") / 255.0

data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, stratify = labels, random_state = 42)

aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.15, horizontal_flip = True, fill_mode = "nearest")

model = GestureNet.build(64, 64, 1, len(lb.classes_))

opt = Adam(lr = conf["init_lr"], decay = conf["init_lr"] / conf["num_epochs"])

model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

H = model.fit_generator(aug.flow(trainX, trainY, batch_size = conf["bs"]), validation_data = (testX, testY), steps_per_epoch = len(trainX) // conf["bs"], epochs = conf["num_epochs"])

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = conf["bs"])
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = lb.classes_))

print("[INFO] saving model...")
model.save(str(conf["model_path"]))

print("[INFO] serializing label encoder...")
f = open(str(conf["lb_path"]), "wb")
f.write(pickle.dumps(lb))
f.close()