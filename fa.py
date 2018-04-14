import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import dlib
import os
import re
from models.mtcnn.align_dlib import AlignDlib
from models.mtcnn import detect_face
from scipy import misc
import csv

align = AlignDlib('models/dlib/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

def detect_face_dlib(img):
    bbs = detector(img, 1)
    tuples = []
    for r in bbs:
        tuples.append((r.left(), r.top(), r.right(), r.bottom()))
    return tuples

EXPECT_SIZE = 160
def align_face_dlib(image, face_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE):
    assert isinstance(face_box, tuple)
    face_rect = dlib.rectangle(*face_box)
    landmarks = align.findLandmarks(image, face_rect)
    alignedFace = align.align(EXPECT_SIZE, image, face_rect,
                              landmarks=landmarks,
                              landmarkIndices=landmarkIndices)
    return alignedFace, landmarks

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

def detect_face_and_landmarks_mtcnn(img):
    img = img[:,:,0:3]
    bbs, lms = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    boxes = []
    landmarks = []
    face_index = 0
    for r in bbs:
        r = r.astype(int)
        points = []
        for i in range(5):
            points.append((lms[i][face_index] , lms[i+5][face_index]))
        landmarks.append(points)
        boxes.append((r[0] , r[1] , r[2] , r[3]))
        #boxes.append(r[:4].astype(int).tolist())
        face_index += 1
    return boxes, landmarks

EXPECT_SIZE = 160
def align_face_mtcnn(img, bb, landmarks):
    assert isinstance(bb, tuple)
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    scaled = misc.imresize(cropped, (EXPECT_SIZE, EXPECT_SIZE), interp='bilinear')
    return scaled


def draw_rects(image, rects):
    result = image.copy()
    for left, top, right, bottom in rects:
        cv2.rectangle(result, (left, top), (right, bottom), (0, 255, 0), 2)
    return result

def draw_landmarks(image, points):
    result = image.copy()
    for point in points:
        cv2.circle(result, point, 3, (0, 255, 0), -1 )
    return result

gesamt = []
for i in range (10):
    camera = cv2.VideoCapture(0)
    return_value,frame = camera.read()
    camera.release()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


    bbs, lm = detect_face_and_landmarks_mtcnn(img)

    aligned_face, lm = align_face_dlib(img, bbs[0], AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    gesamt.append(lm)
with open('landmarks.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in gesamt:
        writer.writerow(i)