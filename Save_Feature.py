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
        
prototxt='doc/deploy_squeezenet.prototxt'
caffe_model='doc/squeezenet.caffemodel'
mean_file='doc/mean_squeezenet.npy'
caffe.set_mode_gpu()
net = caffe.Net(prototxt,caffe_model,caffe.TEST)
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

def saveFeat(image):
    global prob
    im = caffe.io.resize_image(image,(227,227,3))
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #data blob 结构（n, k, h, w)
    transformer.set_transpose('data', (2, 0, 1)) #改变图片维度顺序，(h, w, k) -> (k, h, w)
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
#    transformer.set_channel_swap('data', (2, 1, 0))

    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    net.forward()

#    labels_filename='card/words_card.txt'
#    labels = np.loadtxt(labels_filename, str, delimiter='\t')
#    prob = net.blobs['prob'].data[0].flatten()
#    order = prob.argsort()[-1]
#    print 'class:', labels[order], 'accuracy: ', prob[order]

    conv1_data = net.blobs['conv10'].data[0] #提取特征
#    np.savetxt(claPath+'feat.txt', conv1_data)
#    np.save(claPath+'feat.npy', conv1_data)
    conv1_data.tofile(claPath+'feat.bin')
    show(conv1_data)

#    conv1_data = net.blobs['fire9/concat'].data[0] #提取特征
#    show(conv1_data)
#
#    conv1_data = net.blobs['fire9/expand3x3'].data[0] #提取特征
#    show(conv1_data)
#
#    conv1_data = net.blobs['fire9/expand1x1'].data[0] #提取特征
#    show(conv1_data)
#
#    conv1_data = net.blobs['fire9/squeeze1x1_fire9/relu_squeeze1x1_0_split_1'].data[0] #提取特征
#    show(conv1_data)
#
#    conv1_data = net.blobs['fire9/squeeze1x1_fire9/relu_squeeze1x1_0_split_0'].data[0] #提取特征
#    show(conv1_data)
#
#    conv1_data = net.blobs['fire9/squeeze1x1'].data[0] #提取特征
#    show(conv1_data)

c = cv2.VideoCapture(0)
while 1:
    ret, image = c.read()
    cv2.rectangle(image,(117,37),(522,442),(0,255,0),2)
    cv2.imshow("aaa", image)
    key = cv2.waitKey(10)
    if key == ord(' '):
        cla = str(raw_input("Please enter class name: "))
        claPath = os.path.join(r'features/%s/' %cla)
        if not os.path.exists(claPath):
            os.makedirs(claPath)
        else:
            print "This class has been saved before"
            os._exit(1)
        img = image[40:440, 120:520]
        img = skimage.img_as_float(image[40:440, 120:520]).astype(np.float32)
        saveFeat(img)
    elif key == 27:
        cv2.destroyAllWindows()
        break