'''
# -*- coding: utf-8 -*-



import cv2
import numpy as np
from Sketcher import Sketcher
from libs.util import MaskGenerator, ImageChunker
from libs.pconv_model import PConvUnet
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
from copy import deepcopy

print('load model...')
model = PConvUnet(vgg_weights=None, inference_only=True)
model.load('pconv_imagenet.h5', train_bn=False)
# model.summary()

img = cv2.imread('img/123.jpg', cv2.IMREAD_COLOR)

img_masked = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)

#sketcher = Sketcher('image', [img_masked, mask], lambda : ((255, 255, 255), 255))
chunker = ImageChunker(512, 512, 30)

cv2.rectangle(mask, pt1=(100,240), pt2=(150,300), color=(255, 255, 255), thickness=-1)  #이걸 바꿔야겠네

cv2.imshow('input_mask', mask)

input_img = img_masked.copy()
cv2.rectangle(input_img, pt1=(100,240), pt2=(150,300), color=(255, 255, 255), thickness=-1)  #이걸 바꿔야겠네
input_img = input_img.astype(np.float32) / 255.

input_mask = cv2.bitwise_not(mask)
input_mask = input_mask.astype(np.float32) / 255.
input_mask = np.repeat(np.expand_dims(input_mask, axis=-1), repeats=3, axis=-1) #여기가 핵심

cv2.imshow('input_img', input_img)
cv2.imshow('input_mask2', input_mask)

print(input_img[200,200])   # [0.83137256  0.82352942  0.78431374]
print(input_mask[200,200])  # 0 0 0
print(input_mask[100,100])  # 1 1 1

print('processing...')

chunked_imgs = chunker.dimension_preprocess(deepcopy(input_img))
chunked_masks = chunker.dimension_preprocess(deepcopy(input_mask))


print('-=-----------chunk--------')
#print(chunked_imgs[200,200],chunked_imgs[250,250])
print(chunked_imgs[0][200][200]
      ,chunked_imgs[0][100][100])  #[  1.  1.  1.] [ 0.88235295  0.86274511  0.86666667]
print(chunked_imgs[0][250][250])    #[ 1.  1.  1.]
print(chunked_masks[0][200][200],chunked_masks[0][100][100])  # [ 0.  0.  0.] [ 1.  1.  1.]
print(chunked_masks[0][250][250])   #[ 0.  0.  0.]
print()


    #for i, im in enumerate(chunked_imgs):
    #  cv2.imshow('im %s' % i, im)
    #  cv2.imshow('mk %s' % i, chunked_masks[i])

pred_imgs = model.predict([chunked_imgs, chunked_masks])
result_img = chunker.dimension_postprocess(pred_imgs, input_img)

print('-=-------------------')
print(input_mask[200,200])  #0 0 0
print(input_mask[100,100])  # 1 1 1
print(result_img[200,200])  # [ 0.85522771  0.82138848  0.76987177]


print('completed!')

cv2.imshow('result', result_img)
print(result_img[200,200])  #[ 0.85522771  0.82138848  0.76987177]
print(result_img[100,100])  #[ 0.88413823  0.86405957  0.87151933]
print(result_img[250,250])  #[ 0.03481266  0.59994566  0.48816139]
print(result_img[350,350])  #[ 0.01424918  0.00835248  0.00053383]

cv2.waitKey(0)




'''
import cv2
import numpy as np

from Sketcher import Sketcher
from libs.util import MaskGenerator, ImageChunker
from libs.pconv_model import PConvUnet
from copy import deepcopy

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
from copy import deepcopy

print('load model...')
model = PConvUnet(vgg_weights=None, inference_only=True)
model.load('pconv_imagenet.h5', train_bn=False)
# model.summary()

img = cv2.imread('img/234.jpg', cv2.IMREAD_COLOR)

img_masked = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)

sketcher = Sketcher('image', [img_masked, mask], lambda : ((255, 255, 255), 255))
chunker = ImageChunker(512, 512, 30)

while True:
  key = cv2.waitKey()

  if key == ord('q'): # quit
    break
  if key == ord('r'): # reset
    print('reset')
    img_masked[:] = img
    mask[:] = 0
    sketcher.show()
  if key == 32: # hit spacebar to run inpainting

    cv2.imshow('input_mask', mask)

    input_img = img_masked.copy()
    input_img = input_img.astype(np.float32) / 255.

    input_mask = cv2.bitwise_not(mask)
    input_mask = input_mask.astype(np.float32) / 255.
    input_mask = np.repeat(np.expand_dims(input_mask, axis=-1), repeats=3, axis=-1) #여기가 핵심

    cv2.imshow('input_img', input_img)
    cv2.imshow('input_mask', input_mask)

    #print(input_img[200, 200])  # [ 1 1 1] <- diff!
    #print(input_mask[200, 200])  # [0 0 0]
    #print(input_mask[100, 100])  # 1 1 1

    print('processing...')

    chunked_imgs = chunker.dimension_preprocess(deepcopy(input_img))
    chunked_masks = chunker.dimension_preprocess(deepcopy(input_mask))

    #print('-=-----------chunk--------')
    # print(chunked_imgs[200,200],chunked_imgs[250,250])
    #print(chunked_imgs[0][200][200], chunked_imgs[0][100][100])  # [  1.  1.  1.] [ 0.88235295  0.86274511  0.86666667]
    #print(chunked_imgs[0][250][250])  # [ 1.  1.  1.]
    #print(chunked_masks[0][200][200], chunked_masks[0][100][100])  # [ 0.  0.  0.] [ 1.  1.  1.]
    #print(chunked_masks[0][250][250])  # [ 0.  0.  0.]
    #print()

    #for i, im in enumerate(chunked_imgs):
    #  cv2.imshow('im %s' % i, im)
    #  cv2.imshow('mk %s' % i, chunked_masks[i])

    pred_imgs = model.predict([chunked_imgs, chunked_masks])
    result_img = chunker.dimension_postprocess(pred_imgs, input_img)

    print('completed!')

    cv2.imshow('result', result_img)

    #print(result_img[200, 200])  # [ 0.85522771  0.82138848  0.76987177]
    #print(result_img[100, 100])  # [ 0.88413823  0.86405957  0.87151933]
    #print(result_img[250, 250])  # [ 0.03481266  0.59994566  0.48816139]
    #print(result_img[350, 350])  # [ 0.01424918  0.00835248  0.00053383]

cv2.destroyAllWindows()


# -*- coding: utf-8 -*-

