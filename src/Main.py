# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 01:15:03 2019

@author: Mango
"""
import sys 
import os
import cv2
import numpy as np
sys.path.append('C:/Users/Mango/Download/Anaconda3/envs/opencv-env/Lib/site-packages')
#from PIL import Image
#sys.path.append(os.path.dirname(os.path.realpath("ChestXrayProject")))
#sys.path.append('C:/Users/Mango/Documents/Python Scripts')
'''
dirname, basename = os.path.split('C:/Users/Mango/Documents/Python Scripts/Predict.py')
sys.path.append(dirname)
module_name = os.path.splitext(basename)[0]
module = importlib.import_module(module_name)
'''
import Predict



Predict = Predict.Predict()
if __name__=="__main__":
    value = (input("enter input here :")).replace("'", "")
    value = Predict.main(value)
    #print(value)