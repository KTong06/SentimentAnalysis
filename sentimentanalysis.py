# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:17:26 2022

@author: KTong
"""
import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,GRU,Bidirectional,Embedding
from tensorflow.keras.utils import plot_model


# STATICS
CSV_URL='https://github.com/Ankit152/IMDB-sentiment-analysis/blob/master/IMDB-Dataset.csv?raw=true'
OHE_PKL_PATH=os.path.join(os.getcwd(),'ohe.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model.h5')
TOKENIZER_PATH=os.path.join(os.getcwd(),'tokenizer_sentiment.json')




# DATA LOADING
df=pd.read_csv(CSV_URL)
# df_backup=df.copy()
# df=df_backup

# DATA INSPECTION
df.head()

# DATA CLEANING
df=df.drop_duplicates()

# REMOVE HTML TAGS
# ? DONT BE GREEDY
# * ZERO OR MORE OCCURENCES
# . ANY CHARACTER EXCEPT FOR NEW LINE(/n)

review=df['review'].values # Features : X
sentiment=df['sentiment'].values # sentiment : Y

for index,rev in enumerate(review):
    review[index]=re.sub('<.*?>',' ',rev)

    # CONVERT TO LOWER CASE AND REMOVE NUMBERS 
    # ^ MEANS NOT
    review[index]=re.sub('[^a-zA-Z]',' ',rev).lower().split()

# FEATURE SELECTION

#%% PREPROCESSING
#     TOKENISATION
vocab_size=10000
oov_token='OOV'

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review)
word_index=tokenizer.word_index
print(word_index)

train_sequences=tokenizer.texts_to_sequences(review)


# PADDING AND TRUNCATING
length_of_review=[len(i) for i in train_sequences] # get len of all rows
print(np.median(length_of_review)) # to get no. of max length for padding

max_len=180

padded_review=pad_sequences(train_sequences,maxlen=max_len,truncating='post',padding='post')


#     ONE HOT ENCODING FOR TARGET
ohe=OneHotEncoder(sparse=False)
sentiment=ohe.fit_transform(np.expand_dims(sentiment,axis=-1))

with open(OHE_PKL_PATH,'wb') as file:
    pickle.dump(ohe,file)



#     TRAIN TEST SPLIT
x_train,x_test,y_train,y_test=train_test_split(padded_review,sentiment,test_size=0.3,random_state=123)

#%% MODEL DEVELOPMENT
embedding_dim=64

model=Sequential()    
model.add(Input(shape=(np.shape(x_train)[1])))
model.add(Embedding(vocab_size,embedding_dim))
model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))    
# model.add(LSTM(128, return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(np.shape(sentiment)[1], activation='softmax'))

model.summary()
plot_model(model,show_shapes=(True))

model.compile(optimizer='adam',loss=('categorical_crossentropy'),metrics=['acc'])

x_train=np.expand_dims(x_train,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)

hist=model.fit(x_train,y_train,batch_size=128,epochs=10,validation_data=(x_test,y_test))

#%% MODEL EVALUATION
hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'])
plt.show()

plt.figure()
plt.plot(hist.history['acc'])
plt.show()

plt.figure()
plt.plot(hist.history['val_loss'])
plt.show()

plt.figure()
plt.plot(hist.history['val_acc'])
plt.show()

plt.figure()
plt.plot(hist.history['loss'],'r--')
plt.plot(hist.history['acc'],'b--')
plt.plot(hist.history['val_loss'],'r')
plt.plot(hist.history['val_acc'],'b')
plt.legend(['loss','acc','val_loss','val_acc'])
plt.show()



test_result=model.evaluate(x_test,y_test)
print(test_result) #loss, acc

y_pred=np.argmax(model.predict(x_test),axis=1)
y_true=np.argmax(y_test,axis=1)

cm=confusion_matrix(y_true, y_pred)
print(cm)
matrix_display=ConfusionMatrixDisplay(cm,display_labels=['positive','negative'])
matrix_display.plot(cmap=plt.cm.Reds)
plt.show()

cr=classification_report(y_true, y_pred)
print(cr)


#%% SAVE MODEL
model.save(MODEL_SAVE_PATH)

token_json=tokenizer.to_json()
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

#%% DISCUSSION/REPORTING
# MODEL ACHIEVE AROUND 84% ACC DURING TRAINING
# RECALL AND F1 SCORE REPORTS 87% AND 84% RESPECTIVELY
# HOWEVER THE MODEL STARTED TO OVERFIT AFTER 2ND EPOCH
# GIVE SUGGESTION: EARLY STOPPING CAN BE INTRODUCED TO AVOID OVERFIT
# INCREASE DROPOUT RATE TO CONTROL OVERFIT
# MAY UTILISE/REFER OTHER ARCHITECTURE OF PRETRAINED MODEL TO TEST
# SUCH AS BERT MODEL, TRANSFORMER MODEL, GPT3 MODEL


#%% LIBRARIES TO CHECK OUT
# CORPUS
# NLTK
# EMBEDDING - TENSORFLOW USE THIS
# 2 TYPES: WORD2VEC, WORD EMBEDDING
# WORD EMBEDDING AKA "FEATURE SELECTION" FOR WORDS
# WORD EMBEDDING HAS ALOT OF WAYS: all of which involve reducing dimension
#     WORD EMBEDDING    
#     WORD2VEC
#     GLOVE
#     LEMINITISATION
#     STEMMING
#     TF-IFD
#     CNN
# EMBED - REDUCING DIMENSION BY INTEGRATING 2 OR MORE ITEMS
# TF-IFD COMPARES THINGS AND EXTRACT UNIQUE/DIFFERENCES, REMOVES SIMILARITIES
# CNN RECOGNISES AND EXTRACT IMPORTANT FEATURES/PATTERN IN SOMETHING
# CAN BE PARAGRAPH OF WORDS OR IMAGES






