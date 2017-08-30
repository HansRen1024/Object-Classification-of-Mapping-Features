#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:30:26 2017

@author: hans

http://blog.csdn.net/renhanchi
"""

import cv2
import os

index = 0
num = 0
c = cv2.VideoCapture(0)
while 1:
    ret, image = c.read()
    cv2.rectangle(image,(117,37),(522,442),(0,255,0),2)
    cv2.imshow("aaa", image)
    key = cv2.waitKey(10)
    if key == ord(' '):
        img = image[40:440, 120:520]
        if not os.path.exists('image/%s/'%index):
            os.makedirs('image/%s/'%index)
        cv2.imwrite('image/%s/%s.jpg'%(index,num), img)
        print "saved image/%s/%s.jpg"%(index,num)
        num += 1
        if num == 3:
            num = 0
            index += 1
    elif key == 27:
        cv2.destroyAllWindows()
        break