'''
Using Correlation Trackers in Dlib, you can track any object in a video stream without needing to train a custom object detector.
Check out the tutorial at: http://www.codesofinterest.com/2018/02/track-any-object-in-video-with-dlib.html
'''
import numpy as np
import cv2
import dlib
from Sketcher2 import Sketcher
import os
import tensorflow as tf
from copy import deepcopy
from libs.util import MaskGenerator, ImageChunker
from libs.pconv_model import PConvUnet

from copy import deepcopy
import sys

model = PConvUnet(vgg_weights=None, inference_only=True)
model.load('pconv_imagenet.h5', train_bn=False)
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
chunker = ImageChunker(512, 512, 3)
descs = np.load('img/descs.npy')[()]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(allow_growth=True)

video_path = "img/mk.mp4"

# this variable will hold the coordinates of the mouse click events.
mousePoints = []

# setup flag
def mouseEventHandler(event, x, y, flags, param):
    # references to the global mousePoints variable
    global mousePoints

    # if the left mouse button was clicked, record the starting coordinates.
    if event == cv2.EVENT_LBUTTONDOWN:
        mousePoints = [(x, y)]

    # when the left mouse button is released, record the ending coordinates.
    elif event == cv2.EVENT_LBUTTONUP:
        mousePoints.append((x, y))

# create the video capture.
video_capture = cv2.VideoCapture(0)

# initialize the correlation tracker.
tracker = dlib.correlation_tracker()

# this is the variable indicating whether to track the object or not.
tracked = False

cap = cv2.VideoCapture(video_path)
ret, image = cap.read()
img_masked = image.copy()
mask = np.zeros(image.shape[:2], np.uint8)
point = []
sketcher=Sketcher('image', [img_masked, mask], lambda : ((255, 255, 255), 255), point)
cap.release()

# saving all points and find largest rectangle
while True:
    key = cv2.waitKey()
    if key == 32:
        print('processing')
        np.save("point", point)
        print("point :", point)
        break
max_x = 0
max_y = 0
min_x = 9999
min_y = 9999
for pt in point:
    if max_x < pt[0] : max_x = pt[0]
    if max_y < pt[1] : max_y = pt[1]
    if min_x > pt[0] : min_x = pt[0]
    if min_y > pt[1] : min_y = pt[1]

print((max_x, max_y), (min_x, min_y))
mousePoints = []
mousePoints.append((min_x, min_y))
mousePoints.append((max_x, max_y))

prex = (mousePoints[1][0] + mousePoints[0][0])/2
prey = (mousePoints[1][1] + mousePoints[0][1])/2

cap = cv2.VideoCapture(video_path)

padding_size = 0
resized_width = 320
video_size = (resized_width, resized_width)
output_size = (resized_width, resized_width)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS),output_size)

while(cap.isOpened()):
    # start capturing the video stream.
    ret, frame = cap.read()

    # output size

    #frame = cv2.resize(frame, output_size)

    mask = np.zeros(image.shape[:2], np.uint8)  # mask 선언

    if ret:
        image = frame

        # input polylines
        '''
        pts = np.array(point, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (255, 255, 255), 5)
        '''
        # if we have two sets of coordinates from the mouse event, draw a rectangle.
        if len(mousePoints) == 2:
            #cv2.rectangle(image, mousePoints[0], mousePoints[1], (0, 255, 0), 2)
            pts = np.array(point, np.int32)
            pts = pts.reshape((-1, 1, 2))
            poly = cv2.polylines(image, [pts], True, (255, 255, 255), 10)
            dlib_rect = dlib.rectangle(mousePoints[0][0], mousePoints[0][1], mousePoints[1][0], mousePoints[1][1])

        # tracking in progress, update the correlation tracker and get the object position.
        if tracked == True:
            tracker.update(image)
            track_rect = tracker.get_position()
            x  = int(track_rect.left())
            y  = int(track_rect.top())
            x1 = int(track_rect.right())
            y1 = int(track_rect.bottom())
            difx = (x1+x)/2 - prex
            dify = (y1+y)/2 - prey
            prex = (x1+x)/2
            prey = (y1+y)/2

            for pt in point:
                pt[0] += difx/2
                pt[1] += dify/2
            pts = np.array(point, np.int32)
            pts = pts.reshape((-1, 1, 2))
            poly = cv2.polylines(image, [pts], True, (255, 255, 255), 10)
            cv2.polylines(mask, [pts], True, (255, 255, 255), 10)#123
            #cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)


        # Processing...
        input_img = frame.copy()

        # input_img = img_masked.copy()
        input_img = input_img.astype(np.float32) / 255.

        input_mask = cv2.bitwise_not(mask)
        input_mask = input_mask.astype(np.float32) / 255.
        input_mask = np.repeat(np.expand_dims(input_mask, axis=-1), repeats=3, axis=-1)  # 여기가 핵심

        chunked_imgs = chunker.dimension_preprocess(deepcopy(input_img))
        chunked_masks = chunker.dimension_preprocess(deepcopy(input_mask))

        pred_imgs = model.predict([chunked_imgs, chunked_masks])
        result_img = chunker.dimension_postprocess(pred_imgs, input_img)

        result_img = np.uint8(255 * result_img)
        writer.write(result_img)

        # show the current frame.
        cv2.imshow('video2', result_img)

        if cv2.waitKey(1) == ord('q'):
            break

    # capture the keyboard event in the OpenCV window.
    ch = 0xFF & cv2.waitKey(1)

    # press "r" to stop tracking and reset the points.
    if ch == ord("r"):
        mousePoints = []
        tracked = False

    # start tracking the currently selected object/area.
    if len(mousePoints) == 2:
        #tracker.start_track(image, dlib_rect)
        tracker.start_track(image, dlib_rect)
        tracked = True
        mousePoints = []

    # press "q" to quit the program.
    if ch == ord('q'):
        break

# cleanup.
cap.release()
cv2.destroyAllWindows()
writer.release()
