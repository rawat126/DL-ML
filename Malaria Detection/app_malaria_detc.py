# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:53:43 2020

@author: Ragvender Rawat
"""
import streamlit as st
import numpy as np
from PIL import Image
import pickle as pk
#import tensorflow as tf
import keras
    

def predict_catagory(img, models):
    img = cv2.resize(img,(100,100,3))
    img = np.array([img])
    output = [model.predict(img) for model in models]
    output = np.divide(np.sum(output[0],output[1]),2)
    
    print(output)
    st.write(output)
                
def main():
    with open(r'malaria_model.pkl','rb') as ff_1:
        custom_model = pk.load(ff_1)
    
    with open(r'malaria_VGGmodel.pkl','rb') as ff_2:
        vgg16_model = pk.load(ff_2)
        
    models = [custom_model,vgg16_model]    
    
    st.title('''
             MALARIA DETECTION USING CNN
             ''')
    st.text('''
            We looked at easy to build open-source techniques leveraging AI 
            which can give us state-of-the-art accuracy in detecting malaria 
            thus enabling AI for social good.
            ''')
            
    
    img_file = st.file_uploader(' Please Select Your Image ', 
                     type = ['jpg','png','bmp','tiff','gif','jpeg'])
    
    try:
        image = Image.open(img_file)
        img_data = np.asarray(image, dtype = 'uint8')
        st.image(image, channel = 'BGR', caption = 'Blood Sample of Subject ')
        
        if st.button('Predict'):
            predict_catagory(img_data, models)
            pass
        
    
    except AttributeError:
        st.text('')

main()