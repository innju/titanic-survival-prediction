# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 22:52:17 2022

This script is used to build the simple web app using streamlit.

@author: User
"""

# packages
import os
import pickle
import numpy as np
import streamlit as st

#%% Static Code
# scaler saved
MMS_PATH = os.path.join(os.getcwd(),'mms_scaler.pkl')
# model saved
MODEL_PATH = os.path.join(os.getcwd(),'titanic_model.pkl')

#%% Load pickle files
# load model
model = pickle.load(open(MODEL_PATH,'rb'))
# load scaler
scaler = pickle.load(open(MMS_PATH,'rb'))

#%% build app using streamlit
survival_chance = {0:'Not survive',1:'Survive'}

# create the form
with st.form('Titanic survival'):
    st.write("Passenger's info")
    #features selected 
    Pclass = int(st.number_input('Pclass(Social-economic status):1-Upper,2-Middle,3-Lower'))
    Sex = int(st.number_input('Sex:0-Female,1-Male'))
    Age = st.number_input('Age')
    Familysize = int(st.number_input('Family size:Total no. of siblings,spouses,parents and children abroad'))
    Fare = st.number_input('Fare')
    Embarked = int(st.number_input('Embarked(Port of Embarkation):0-Cherbourg,1-Queenstown,2-Southampton'))
    
    submitted= st.form_submit_button('Submit')
    
    
    # to observe if the information appear if i click submit
    if submitted == True:
        
        passenger_info = np.array([Pclass,Sex,Age,Familysize,Fare,Embarked])
        
        info_scaled = scaler.transform(np.expand_dims(passenger_info, axis=0))
        
        outcome= model.predict(info_scaled)
        
        if outcome == 1:
            st.success('Survive')
        else:
            st.snow() # effect of snow
            st.warning('Not survive')
            

