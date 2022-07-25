# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:34:26 2022
Author: danny.rashd
Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
"""
# 1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,datetime
#%%
# 2. Load data from csv
data = pd.read_csv('dataset/heart.csv')
data.head()
#%%
# 3. Check for missing values
data.isna().sum()
#%%
labels = data['target']
features = data.drop('target',axis=1)
print("Heart Disease dataset has {} data points with {} variables each.".format(*data.shape))

#%%
print('===================Features===================')
print(features.head())
print('===================Labels===================')
print(labels.head())
#%%
# Train test split
SEED= 12345
X_train, X_test, y_train,y_test= train_test_split(features,labels,test_size=0.2,random_state=SEED)
#%%
# Feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test = scaler.transform(X_test)

#%%
# Create sequential network using keras
nClass = len(np.unique(y_test))
nIn = X_test.shape[1]

inputs = keras.Input(shape=(nIn,))

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(nIn,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(8,activation='relu'))
model.add(layers.Dense(nClass, activation='softmax'))
dot_img_file = 'images/model_1.png'
plot_model(model, to_file=dot_img_file, show_shapes=True)
model.summary()
#%%
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#%%
# Initialize callbacks

# EarlyStopping 
es = EarlyStopping(patience=5,verbose=1,restore_best_weights=True)

# TensorBoard 
PATH = "tb_logs"
LOG_PATH = os.path.join(PATH,'heart_disease',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb= TensorBoard(log_dir=LOG_PATH)
#%%
# Model training
BATCH_SIZE = 16
EPOCHS= 50
history = model.fit(X_train, y_train, validation_data=(X_test,y_test),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[es,tb])
#%%
# Model Evaluation
# Train evaluation loss and accuracy
print(f"Train Evaluation : \n { model.evaluate(X_train,y_train)}")

# Test evaluation loss and accuracy
print(f"Test Evaluation : \n { model.evaluate(X_test,y_test)}")
#%%
# 8. Visualize train and test results
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epoch_x = history.epoch

plt.plot(epoch_x,train_loss, label='Training Loss')
plt.plot(epoch_x, val_loss, label='Validation Loss')
plt.title('Training Loss vs Validation Loss')
plt.legend()
plt.figure()

plt.plot(epoch_x, train_acc,label='Train Accuracy')
plt.plot(epoch_x, val_acc, label='Validation Accuracy')
plt.title('Training Accuracy vs Validation Accuracy')
plt.legend()
plt.figure()

plt.show()