#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:05:13 2017

@author: hans

http://blog.csdn.net/renhanchi
"""

import skimage
import caffe
import cv2
import os 
import numpy as np
from math import sqrt
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

conv_name = 'loss3/classifier'
net_mode = '_googlenet'
prototxt='doc/deploy_googlenet.prototxt'
caffe_model='doc/googlenet.caffemodel'
mean_file='doc/mean_googlenet.npy'

#conv_name = 'pool10'
#net_mode = '_squeezenet'
#prototxt='doc/deploy_squeezenet.prototxt'
#caffe_model='doc/squeezenet.caffemodel'
#mean_file='doc/mean_squeezenet.npy'

dirpath = 'features/'
caffe.set_mode_gpu()
net = caffe.Net(prototxt,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #data blob 结构（n, k, h, w)
transformer.set_transpose('data', (2, 0, 1)) #改变图片维度顺序，(h, w, k) -> (k, h, w)
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
#transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR，Opencv读取的图片通道已经是BGR的

def contrastFeat(image):
    similarity = []
    cla = []
    L2_list = []
    im = caffe.io.resize_image(image,(224,224,3))
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    net.forward()
    conv1_data = net.blobs[conv_name].data[0] #提取特征
    
    for dirname in os.listdir(dirpath):
        if os.path.isdir(r'%s%s/' %(dirpath, dirname)):
            for i in range(3):
                claPath = os.path.join(r'%s%s/' %(dirpath, dirname))
    #            feat = np.load(claPath+'feat.npy')
                feat = np.fromfile(claPath+'loss3_classifier'+net_mode+'_%s.bin'%i, dtype = np.float32)
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
    for i in range(len(L2_list)):
        similarity.append(L2_list[i]/sum(L2_list))
    similarity = np.array(similarity)
    order = similarity.argsort()[0]
    return int(cla[order])

data = []
labels = []
name_list = []
with open('doc/card_test.txt','r') as f:
    while 1:
        line = f.readline()
        if line:
            line=line.strip()
            p=line.rfind(' ')
            data.append(line[0:p])
            labels.append(int(line[p+1::]))
        else:
            break
correct = 0
for i in range(len(data)):
    print i,'/',len(data),' processing... ',data[i]
    image = cv2.imread(data[i])
    label = labels[i]
    img = skimage.img_as_float(image).astype(np.float32)
    predict = contrastFeat(img)
    if predict == label:
        correct += 1
    else:
        name_list.append(data[i].split('/')[-1])
    accuracy = float(correct)/float(i+1)
print "Total: ",len(data),"Correct: ",correct,"Accuracy: ",accuracy
