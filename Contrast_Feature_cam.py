#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:59:44 2017

@author: hans

http://blog.csdn.net/renhanchi
"""

import caffe
import time
import cv2
import os 
import skimage
import numpy as np
from math import sqrt
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


#conv_name = 'loss3/classifier'
#net_mode = '_googlenet'
#prototxt='doc/deploy_googlenet.prototxt'
#caffe_model='doc/googlenet.caffemodel'
#mean_file='doc/mean_googlenet.npy'


conv_name = 'pool10'
net_mode = '_squeezenet'
prototxt='doc/deploy_squeezenet.prototxt'
caffe_model='doc/squeezenet.caffemodel'
mean_file='doc/mean_squeezenet.npy'

dirpath = 'features/'
caffe.set_mode_gpu()
net = caffe.Net(prototxt,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #data blob 结构（n, k, h, w)
transformer.set_transpose('data', (2, 0, 1)) #改变图片维度顺序，(h, w, k) -> (k, h, w)
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
#transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR

clas = []
with open('doc/words_card.txt','r') as f:
    while 1:
        line = f.readline()
        if line:
            line=line.strip()
            p=line.rfind(' ')
            clas.append(line[p+1::])
        else:
            break

def contrastFeat(image):
    t0 = time.time()
    global L2_list
    global cla
    global order
    cla = []
    L2_list = []
    im = caffe.io.resize_image(image,(227,227,3))
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    net.forward()
    conv1_data = net.blobs[conv_name].data[0] #提取特征
    
    for dirname in os.listdir(dirpath):
        if os.path.isdir(r'%s%s/' %(dirpath, dirname)):
            for i in range(3):
                claPath = os.path.join(r'%s%s/' %(dirpath, dirname))
    #            feat = np.load(claPath+'feat.npy')
                feat = np.fromfile(claPath+'pool10'+net_mode+'_%s.bin'%i, dtype = np.float32)
                feat = feat.reshape(conv1_data.shape)
                dis = 0
                for n in range(feat.shape[0]):
                    if len(feat.shape)>1:
                        for h in range(feat.shape[1]):
                            for w in range(feat.shape[2]):
                                dis += pow(conv1_data[n,h,w]-feat[n,h,w],2)
                    else:
                        dis += pow(conv1_data[n]-feat[n],2)
                L2_list.append(sqrt(dis))
                cla.append(dirname)
    L2_list = np.array(L2_list)
    order = L2_list.argsort()[0]
    t1 = time.time()
    print 'order:',cla[order],'clss:', clas[int(cla[order])], 'prob:', L2_list[order],'elapsed:',t1-t0

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
