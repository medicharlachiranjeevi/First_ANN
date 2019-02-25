#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:00:29 2019

@author: system
"""
import pandas as pd
import  keras.utils.np_utils as np_utils
from sklearn.preprocessing import LabelEncoder

def data(file):
# Importing the dataset
 dataset = pd.read_csv(file,header=None)

 X = dataset.iloc[:, 1:11].values
 y = dataset.iloc[:, -1].values
 #imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
 labelencoder_X_0 = LabelEncoder()
 X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0].astype('str'))

 labelencoder_X_1 = LabelEncoder()
 X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1].astype('str'))
 labelencoder_Y_2 = LabelEncoder()
 y_t = labelencoder_Y_2.fit_transform(y.astype('str'))

# encode class values as integers
# convert integers to dummy variables (i.e. one hot encoded)
 y_t = np_utils.to_categorical(y_t)

 labelencoder_X_2 = LabelEncoder()
 X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2].astype('str'))
 labelencoder_X_3 = LabelEncoder()
 X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3].astype('str'))
 labelencoder_X_4 = LabelEncoder()
 X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4].astype('str'))
 labelencoder_X_5 = LabelEncoder()
 X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5].astype('str'))
 labelencoder_X_6 = LabelEncoder()
 X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6].astype('str'))
 labelencoder_X_7 = LabelEncoder()
 X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7].astype('str'))
 labelencoder_X_8 = LabelEncoder()
 X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8].astype('str'))
 labelencoder_X_9 = LabelEncoder()
 X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9].astype('str'))
 return X,y_t
def data_pred(file):
# Importing the dataset
 dataset = pd.read_csv(file)

 X = dataset.iloc[:, 1:11].values
 #imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Encoding categorical data

 labelencoder_X_0 = LabelEncoder()
 X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0].astype('str'))

 labelencoder_X_1 = LabelEncoder()
 X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1].astype('str'))

 labelencoder_X_2 = LabelEncoder()
 X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2].astype('str'))
 labelencoder_X_3 = LabelEncoder()
 X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3].astype('str'))
 labelencoder_X_4 = LabelEncoder()
 X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4].astype('str'))
 labelencoder_X_5 = LabelEncoder()
 X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5].astype('str'))
 labelencoder_X_6 = LabelEncoder()
 X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6].astype('str'))
 labelencoder_X_7 = LabelEncoder()
 X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7].astype('str'))
 labelencoder_X_8 = LabelEncoder()
 X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8].astype('str'))
 labelencoder_X_9 = LabelEncoder()
 X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9].astype('str'))
 return X  
