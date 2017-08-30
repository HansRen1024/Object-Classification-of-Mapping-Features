#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:09:05 2017

@author: hans

http://blog.csdn.net/renhanchi
"""

import caffe
import cv2
import os 
import skimage
import numpy as np
import matplotlib.pyplot as plt
        
conv_name = 'loss3/classifier'
net_mode = '_googlenet'
prototxt='doc/deploy_googlenet.prototxt'
caffe_model='doc/googlenet.caffemodel'
mean_file='doc/mean_googlenet.npy'

caffe.set_mode_gpu()
net = caffe.Net(prototxt,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #data blob 结构（n, k, h, w)
transformer.set_transpose('data', (2, 0, 1)) #改变图片维度顺序，(h, w, k) -> (k, h, w)
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)

for name,feature in net.blobs.items(): #查看各层特征规模
    print name + '\t' + str(feature.data.shape)

def show(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')

def saveFeat(image,num):
    im = caffe.io.resize_image(image,(224,224,3))
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    net.forward()

    conv1_data = net.blobs[conv_name].data[0] #提取特征
    conv1_data.tofile(claPath+'loss3_classifier'+net_mode+'_%s.bin'%num)
    print "saved",claPath+'loss3_classifier'+net_mode+'_%s.bin'%num

#    conv2_data = net.blobs['fire9/concat'].data[0]
#    conv2_data.tofile(claPath+'fire9_concat'+net_mode+'_%s.bin'%num)
#
#    conv3_data = net.blobs['fire9/expand3x3'].data[0]
#    conv3_data.tofile(claPath+'fire9_expand3x3'+net_mode+'_%s.bin'%num)
#
#    conv3_data = net.blobs['fire9/expand1x1'].data[0]
#    conv3_data.tofile(claPath+'fire9_expand1x1'+net_mode+'_%s.bin'%num)

for dirname in os.listdir(r'image/'):
    if os.path.isdir(r'image/%s'%dirname):
        for im_path in os.listdir(r'image/%s'%dirname):
            claPath = os.path.join(r'features/%s/' %dirname)
            img = cv2.imread(r'image/%s/%s'%(dirname,im_path))
            img = skimage.img_as_float(img).astype(np.float32)
            saveFeat(img, im_path.split('.'[0])[0])