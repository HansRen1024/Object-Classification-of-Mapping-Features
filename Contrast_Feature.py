#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:39:58 2017

@author: hans

http://blog.csdn.net/renhanchi
"""

import caffe
import cv2
import os 
import skimage
import numpy as np
from math import sqrt

dirpath = r'features/'
prototxt='deploy_squeezenet.prototxt'
caffe_model='squeezenet.caffemodel'
mean_file='mean_squeezenet.npy'
caffe.set_mode_gpu()
net = caffe.Net(prototxt,caffe_model,caffe.TEST)

def contrastFeat(image):
    global similarity
    similarity = []
    cla = []
    im = caffe.io.resize_image(image,(227,227,3))
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #data blob 结构（n, k, h, w)
    transformer.set_transpose('data', (2, 0, 1)) #改变图片维度顺序，(h, w, k) -> (k, h, w)
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
#    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    net.forward()
    conv1_data = net.blobs['conv10'].data[0] #提取特征
    
    for dirname in os.listdir(dirpath):
        if os.path.isdir(r'%s%s/' %(dirpath, dirname)):
            claPath = os.path.join(r'%s%s/' %(dirpath, dirname))
#            feat = np.load(claPath+'feat.npy')
            feat = np.fromfile(claPath+'feat.bin', dtype = np.float32)
            feat = feat.reshape(conv1_data.shape[0],conv1_data.shape[1],conv1_data.shape[2])
            dis = 0
            for n in range(feat.shape[0]):
                for h in range(feat.shape[1]):
                    for w in range(feat.shape[2]):
                        dis += pow(conv1_data[n,h,w]-feat[n,h,w],2)
            L2 = sqrt(dis)
            similarity.append(1/(1+L2))
            cla.append(dirname)
    similarity = np.array(similarity)
    print similarity
    order = similarity.argsort()[-1]
    print 'clss:', cla[order], 'prob:', similarity[order]

c = cv2.VideoCapture(0)
while 1:
    ret, image = c.read()
    cv2.rectangle(image,(117,37),(522,442),(0,255,0),2)
    cv2.imshow("aaa", image)
    key = cv2.waitKey(10)
    if key == ord(' '):
        img = image[40:440, 120:520]
        img = skimage.img_as_float(image[40:440, 120:520]).astype(np.float32)
        contrastFeat(img)
    elif key == 27:
        cv2.destroyAllWindows()
        break