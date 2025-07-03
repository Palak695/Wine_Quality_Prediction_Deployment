# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 00:02:09 2025

@author: Palak Agrawal
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def wine_quality_prediction(input_data):
    

    # changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the data as we are predicting the label for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==1):
      return 'Good Quality Wine'
    else:
      return 'Bad Quality Wine'

def main():
    
    
    # giving a title
    st.title('Wine Quality Prediction Web App')
    
    
    # getting the input data from the user
    

    
    fixed_acidity = st.sidebar.slider('fixed acidity', 4.6, 15.9, 8.31)
    volatile_acidity = st.sidebar.slider('volatile acidity', 0.12,1.58 , 0.52)
    citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
    chlorides = st.sidebar.slider('chlorides', 0.01,0.6 , 0.08)
    total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,289.0 , 46.0)
    alcohol=st.sidebar.slider('alcohol', 8.4,14.9, 10.4)
    sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.9, 6.0, 2.5)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1, 40, 16)
    density = st.sidebar.slider('Density', 0.9900, 1.0040, 0.9968)
    pH = st.sidebar.slider('pH', 2.9, 3.6, 3.3)
    
    
    # code for Prediction
    quality = ''
    
    # creating a button for Prediction
    
    if st.button('Wine Quality Test Result'):
        quality = wine_quality_prediction([fixed_acidity,volatile_acidity,
                                           citric_acid,residual_sugar,chlorides,
                                           free_sulfur_dioxide,total_sulfur_dioxide,density,
                                           pH,sulphates,alcohol])
        
        
    st.success(quality)
    
    
    
    
    
if __name__ == '__main__':
    main()