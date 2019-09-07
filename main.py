
import dlib, cv2
import numpy as np
import os
import tensorflow as tf
from copy import deepcopy
from libs.util import MaskGenerator, ImageChunker
from libs.pconv_model import PConvUnet

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

# open video file
video_path = 'img/tom2.mp4'
cap = cv2.VideoCapture(video_path)

output_size = (320,320) # (width, height)
fit_to = 'height'

# initialize writing video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

# check file is opened
if not cap.isOpened():
  exit()

# initialize tracker
OPENCV_OBJECT_TRACKERS = {
  "csrt": cv2.TrackerCSRT_create,
  "kcf": cv2.TrackerKCF_create,
  "boosting": cv2.TrackerBoosting_create,
  "mil": cv2.TrackerMIL_create,
  "tld": cv2.TrackerTLD_create,
  "medianflow": cv2.TrackerMedianFlow_create,
  "mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS['csrt']()

# global variables
top_bottom_list, left_right_list = [], []
count = 0

# main
ret, img = cap.read()

cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)

# select ROI
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')

# initialize tracker
tracker.init(img, rect)

while True:
  count += 1
  # read frame from video
  ret, img = cap.read()

  if not ret:
    exit()

  # update tracker and get position from new frame
  success, box = tracker.update(img)
  # if success:
  left, top, w, h = [int(v) for v in box]
  right = left + w
  bottom = top + h

  # save sizes of image
  top_bottom_list.append(np.array([top, bottom]))
  left_right_list.append(np.array([left, right]))

  # use recent 10 elements for crop (window_size=10)
  if len(top_bottom_list) > 10:
    del top_bottom_list[0]
    del left_right_list[0]

  # compute moving average
  avg_height_range = np.mean(top_bottom_list, axis=0).astype(np.int)
  avg_width_range = np.mean(left_right_list, axis=0).astype(np.int)
  avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)]) # (x, y)

  # compute scaled width and height
  scale = 1.3
  avg_height = (avg_height_range[1] - avg_height_range[0]) * scale
  avg_width = (avg_width_range[1] - avg_width_range[0]) * scale

  # compute new scaled ROI
  avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2])
  avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2])

  # fit to output aspect ratio
  if fit_to == 'width':
    avg_height_range = np.array([
      avg_center[1] - avg_width * output_size[1] / output_size[0] / 2,
      avg_center[1] + avg_width * output_size[1] / output_size[0] / 2
    ]).astype(np.int).clip(0, 9999)

    avg_width_range = avg_width_range.astype(np.int).clip(0, 9999)
  elif fit_to == 'height':
    avg_height_range = avg_height_range.astype(np.int).clip(0, 9999)

    avg_width_range = np.array([
      avg_center[0] - avg_height * output_size[0] / output_size[1] / 2,
      avg_center[0] + avg_height * output_size[0] / output_size[1] / 2
    ]).astype(np.int).clip(0, 9999)

  # crop image
  result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

  # resize image to output size
  result_img = cv2.resize(result_img, output_size)
  img = cv2.resize(img, output_size)

  #mask
  mask = np.zeros(img.shape[:2], np.uint8)  # mask 선언


  # visualize
  pt1 = (int(left), int(top))
  pt2 = (int(right), int(bottom))
  cv2.rectangle(img, pt1, pt2, (255, 255, 255), -1) #image
  cv2.rectangle(mask, pt1, pt2, (255, 255, 255), -1)  #masking

  # Processing...
  input_img = img.copy()

  # input_img = img_masked.copy()
  input_img = input_img.astype(np.float32) / 255.

  input_mask = cv2.bitwise_not(mask)
  input_mask = input_mask.astype(np.float32) / 255.
  input_mask = np.repeat(np.expand_dims(input_mask, axis=-1), repeats=3, axis=-1)  # 여기가 핵심

  chunked_imgs = chunker.dimension_preprocess(deepcopy(input_img))
  chunked_masks = chunker.dimension_preprocess(deepcopy(input_mask))

  pred_imgs = model.predict([chunked_imgs, chunked_masks])
  final_img = chunker.dimension_postprocess(pred_imgs, input_img)



  cv2.imshow('img', img)
  cv2.imshow('result', np.uint8(255 * final_img))

  # write video
  out.write(np.uint8(255 * final_img))
  if cv2.waitKey(1) == ord('q'):
    break

# release everything
cap.release()
out.release()
cv2.destroyAllWindows()



'''
import cv2
import numpy as np

# open video file
video_path = 'img/track_test.mp4'
cap = cv2.VideoCapture(video_path)

output_size = (120,200) # (width, height)
fit_to = 'height'

# initialize writing video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

# check file is opened
if not cap.isOpened():
  exit()

# initialize tracker
OPENCV_OBJECT_TRACKERS = {
  "csrt": cv2.TrackerCSRT_create,
  "kcf": cv2.TrackerKCF_create,
  "boosting": cv2.TrackerBoosting_create,
  "mil": cv2.TrackerMIL_create,
  "tld": cv2.TrackerTLD_create,
  "medianflow": cv2.TrackerMedianFlow_create,
  "mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS['csrt']()

# global variables
top_bottom_list, left_right_list = [], []
count = 0

# main
ret, img = cap.read()

cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)

# select ROI
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')

# initialize tracker
tracker.init(img, rect)

while True:
  count += 1
  # read frame from video
  ret, img = cap.read()

  if not ret:
    exit()

  # update tracker and get position from new frame
  success, box = tracker.update(img)
  # if success:
  left, top, w, h = [int(v) for v in box]
  right = left + w
  bottom = top + h

  # save sizes of image
  top_bottom_list.append(np.array([top, bottom]))
  left_right_list.append(np.array([left, right]))

  # use recent 10 elements for crop (window_size=10)
  if len(top_bottom_list) > 10:
    del top_bottom_list[0]
    del left_right_list[0]

  # compute moving average
  avg_height_range = np.mean(top_bottom_list, axis=0).astype(np.int)
  avg_width_range = np.mean(left_right_list, axis=0).astype(np.int)
  avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)]) # (x, y)

  # compute scaled width and height
  scale = 1.3
  avg_height = (avg_height_range[1] - avg_height_range[0]) * scale
  avg_width = (avg_width_range[1] - avg_width_range[0]) * scale

  # compute new scaled ROI
  avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2])
  avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2])

  # fit to output aspect ratio
  if fit_to == 'width':
    avg_height_range = np.array([
      avg_center[1] - avg_width * output_size[1] / output_size[0] / 2,
      avg_center[1] + avg_width * output_size[1] / output_size[0] / 2
    ]).astype(np.int).clip(0, 9999)

    avg_width_range = avg_width_range.astype(np.int).clip(0, 9999)
  elif fit_to == 'height':
    avg_height_range = avg_height_range.astype(np.int).clip(0, 9999)

    avg_width_range = np.array([
      avg_center[0] - avg_height * output_size[0] / output_size[1] / 2,
      avg_center[0] + avg_height * output_size[0] / output_size[1] / 2
    ]).astype(np.int).clip(0, 9999)

  # crop image
  result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

  # resize image to output size
  result_img = cv2.resize(result_img, output_size)

  # visualize
  pt1 = (int(left), int(top))
  pt2 = (int(right), int(bottom))
  cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)

  cv2.imshow('img', img)
  cv2.imshow('result', result_img)
  # write video
  out.write(result_img)
  if cv2.waitKey(1) == ord('q'):
    break

# release everything
cap.release()
out.release()
cv2.destroyAllWindows()
'''