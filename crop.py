# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:43:16 2016

@author: AdminS1003828
"""
import math

def distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(50,50,20,50)):
    dist_basic = 25
    dist = distance(eye_left,eye_right)
    pixels_h_left = offset_pct[0] * dist/dist_basic
    pixels_h_right = offset_pct[1] * dist/dist_basic
    pixels_v_up = offset_pct[2] * dist/dist_basic
    pixels_v_down = offset_pct[3] * dist/dist_basic
    eye_y = (eye_left[1] + eye_right[1])/2
    
    reshapeBox = [int(round(eye_left[0] - pixels_h_left)), int(round(eye_right[0] + pixels_h_right)), int(round(eye_y - pixels_v_up)), int(round(eye_y + pixels_v_down))]
    if(reshapeBox[0] < 0):
        reshapeBox[0] = 0
    if(reshapeBox[1] > image.shape[1]):
        reshapeBox[1] = image.shape[1]
    if(reshapeBox[2] < 0):
        reshapeBox[2] = 0
    if(reshapeBox[3] > image.shape[0]):
        reshapeBox[3] = image.shape[0]
    imageNew = image[reshapeBox[2]:reshapeBox[3],reshapeBox[0]:reshapeBox[1]]
    return imageNew