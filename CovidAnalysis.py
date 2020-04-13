# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:18:52 2020

@author: iftikhar
"""
#importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
os.chdir("E:\cases")
data = pd.read_excel('covid_data.xlsx')
data['Fever'] = data['Fever'].round(3)
print(data.head())
column_list=list(data.columns)
print(column_list)
data.isna().sum()
#visualization
plt.figure(figsize=(10,5))
sns.heatmap(data.corr(), cmap='coolwarm', annot=True, linewidth = 0.5)
#features
features = list(set(column_list)-set(['Infected']))
print(features)
#features and target values
x = data[features].values
y = data['Infected'].values
#train test split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)
#Logistic regression
logistic = LogisticRegression()
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_
prediction = logistic.predict(test_x)
print(prediction)
confusion_matrix = confusion_matrix(test_y,prediction)
print(confusion_matrix)
accuracy = accuracy_score(test_y,prediction)
print(accuracy)
#kNN
kNN_classifier = KNeighborsClassifier(n_neighbors = 5)
kNN_classifier.fit(train_x,train_y)
prediction2 = kNN_classifier.predict(test_x)
accuracy2 = accuracy_score(test_y,prediction2)
print(accuracy2)
#Random forest
RF_model = RandomForestClassifier(n_estimators=500).fit(train_x,train_y)
RF_predictions = RF_model.predict(test_x)
RF_accuracy = accuracy_score(test_y,RF_predictions)
print(RF_accuracy)
#decison tree
DT_model = DecisionTreeClassifier(criterion='entropy').fit(train_x,train_y)
DT_predictions = DT_model.predict(test_x)
DT_accuracy = accuracy_score(test_y,DT_predictions)
print(DT_accuracy)
#naive bayes
NB_model = GaussianNB().fit(train_x,train_y)
NB_predictions = NB_model.predict(test_x)
NB_accuracy = accuracy_score(test_y,NB_predictions)
print(NB_accuracy)