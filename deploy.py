# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:16:57 2022

@author: KTong
"""

import os
import re
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder

#%% STATICS
OHE_PKL_PATH=os.path.join(os.getcwd(),'friend folder','ohe.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'friend folder','model.h5')
TOKENIZER_PATH=os.path.join(os.getcwd(),'friend folder','tokenizer_sentiment.json')


while True:
    
    # To load model
    loaded_model=load_model(os.path.join(os.getcwd(),'model.h5'))
    loaded_model.summary()
    
    # To load tokenizer
    with open(TOKENIZER_PATH,'r') as json_file:
        loaded_tokenizer=json.load(json_file)
    
    # To use JSON format tokenizer 
    tokenizer=tokenizer_from_json(loaded_tokenizer)
    
    # NEW DATA INPUT 
    # input_review='This movie is good good good good good. But actors are @#$%^&!'
    input_review=input('Enter your review:')
    # PREPROCESS DATA
    # Cleaned data from symbols and numbers
    input_review_clean=re.sub('<.*?>',' ',input_review)
    input_review_clean=re.sub('[^a-zA-Z]',' ',input_review).lower().split()
    
    # Encoding and padding review 
    input_review_encoded=tokenizer.texts_to_sequences(input_review_clean)
    ip_review_padded=pad_sequences(np.array(input_review_encoded).T,maxlen=180,truncating='post',padding='post')
    
    # Increase review shape to feed into model
    ip_review_fulldim=np.expand_dims(ip_review_padded,axis=-1) 
    
    # TEST LOADED_MODEL WITH NEW REVIEW
    result=loaded_model.predict(ip_review_fulldim)
    print(result)
    
    # Interpret result
    with open(OHE_PKL_PATH,'rb') as file:
        loaded_ohe=pickle.load(file)
    
    translated_result=loaded_ohe.inverse_transform(result)
    print(translated_result)
