# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:33:28 2019

@author: Mango
"""
import sys 
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import Predict
import keras
from keras.models import load_model
from pathlib import Path
import cv2
import numpy as np
from keras.utils import to_categorical
import h5py
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
# Define path to the data directory
#from PIL import Image
import numpy as np
import os
import cv2
from keras.models import load_model

import sys

class Predict:
    def __init__(self):
        self.init = 0
    def build_model(self):
        input_img = Input(shape=(224,224,3), name='ImageInput')
        x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
        x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
        x = MaxPooling2D((2,2), name='pool1')(x)
        x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
        x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
        x = MaxPooling2D((2,2), name='pool2')(x)
        x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
        x = BatchNormalization(name='bn1')(x)
        x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
        x = MaxPooling2D((2,2), name='pool3')(x)
        x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
        x = BatchNormalization(name='bn3')(x)
        x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
        x = BatchNormalization(name='bn4')(x)
        x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
        x = MaxPooling2D((2,2), name='pool4')(x)
    
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dropout(0.7, name='dropout1')(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='dropout2')(x)
        x = Dense(2, activation='softmax', name='fc3')(x)
        
        model = Model(inputs=input_img, outputs=x)
        f = h5py.File('C:/Users/Mango/Documents/ChestXrayProject/best_model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

# Select the layers for which you want to set weight.

        w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']
        model.layers[1].set_weights = [w,b]

        w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
        model.layers[2].set_weights = [w,b]

        w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']
        model.layers[4].set_weights = [w,b]

        w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']
        model.layers[5].set_weights = [w,b]

        f.close()
        #model.summary()
        return model
    def model(self):
        model =  self.build_model()
        model.summary()
		# opt = RMSprop(lr=0.0001, decay=1e-6)
        opt = Adam(lr=0.0001, decay=1e-5)
        es = EarlyStopping(patience=5)
        chkpt = ModelCheckpoint(filepath='C:/Users/Mango/Documents/ChestXrayProject/best_model/best_model.hdf5', save_best_only=True, save_weights_only=True)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
        model.load_weights("C:/Users/Mango/Documents/ChestXrayProject/best_model/best_model.hdf5")
        return model
    
    def get_cell_name(self, label):
        if label==0:
            return "NORMAL"
        if label==1:
            return "PNEUMONIA"

    def predict(self, img, imgid):
	#print("Predicting Type of Cell Image.................................")
        model1 = self.model()
        imgid = imgid
        img = cv2.imread(str(img))
        #img = Image.open(img)
        #img = np.array(img)
        img = cv2.resize(img, (224,224))
        if img.shape[2] ==1:
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255
        #img=img/255
		#label=1
        result=[]
        a=[]
        a.append(img)
        a=np.array(a)
        score=model1.predict(a,batch_size=16)
        label_index=np.argmax(score, axis=-1)
        acc=np.max(score)
        loss=np.min(score)
        Diag=self.get_cell_name(label_index)        
        result.append(imgid)
        result.append(Diag)
        result.append("Accuracy: "+str(round(acc*100,1))+"%")
        result.append("Loss: "+str(round(loss*100,1))+"%")
        return result
# Get predictions
#pred_p=model.predict(test_data)
# Original labels
    
    def main(self, img):
        imgid = str(img).split("\\")[-1].replace(".jpeg", "")
        print(imgid)
        returnValue=self.predict(img, imgid)
        output = ", ".join(returnValue)
        logout = output.replace("Accuracy:", "").replace("Loss:", "")
        log = open('C:/Users/Mango/Documents/ChestXrayProject/Result/log.csv', 'a')
        log.write("\n" + str(logout))
        result = open("C:/Users/Mango/Documents/ChestXrayProject/Result/lastresult.txt","w") 
        result.write(str(output))
        #print(output)
		#print(file1)
        return output


