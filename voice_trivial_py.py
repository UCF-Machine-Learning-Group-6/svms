# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:47:58 2017

@author: Girish, Sasank, Srikar, Ryan   
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.svm import SVC

df = pd.read_csv('voice.csv')

#Splitting X and Y
X=df.iloc[:, :-1]

y=df.iloc[:,-1]

# Encode label category
# male -> 1
# female -> 0

gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)

# Scale the data to be between -1 and 1

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#Default linear kernel using SVC :
svc=SVC(kernel='linear')
scores = cross_val_score(svc, X, y, cv=5, scoring='accuracy')
print("accuracy for Default Linear kernel using SVC" )
print(scores.mean())

#Default polynomial kernel using SVC :
svc=SVC(kernel='poly')
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') 
print("accuracy for Default Polynomial kernel using SVC" )
print(scores.mean())

#Default Gaussian kernel using SVC :
svc=SVC(kernel='rbf')
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print("accuracy for Default Gaussian kernel using SVC")
print(scores.mean())

print('Accuracies after tuning Hyperparameters:' )

#linear kernel with C value tuning:
C_value = 1
lin_svc = svm.SVC(kernel = 'linear', C = C_value).fit(X, y)
scores = cross_val_score(lin_svc, X, y, cv = 5, scoring = 'accuracy') 
print("linear kernel Accuracy for C = ", C_value)
print(scores.mean())

#polynomial kernel with degree and C values tuning:
C_value = 7.5
degree_value = 3
poly_svc = svm.SVC(kernel='poly', degree = degree_value, C = C_value).fit(X, y)
scores = cross_val_score(poly_svc, X, y, cv = 5, scoring = 'accuracy') 
print("Polynomial kernel Accuracy for degree =", degree_value, "and C =", C_value)
print(scores.mean())

#Gaussian kernel with C and gamma values tuning:
C_value = 1
gamma_value = 0.02
rbf_svc = svm.SVC(kernel='rbf', gamma = gamma_value , C = C_value).fit(X, y)
scores = cross_val_score(rbf_svc, X, y, cv = 5, scoring = 'accuracy') 
print("Gaussian kernel Accuracy for C = ", C_value, "and gamma =", gamma_value)
print(scores.mean())

#Now Checking for a trivial dataset

df = pd.read_csv('trivial.csv')

N=df.iloc[:, :-1]

m=df.iloc[:,-1]

gender_encoder = LabelEncoder()
m = gender_encoder.fit_transform(m)

scaler = StandardScaler()
scaler.fit(N)
N = scaler.transform(N)

#linear kernel with C value tuning:
C_value = 0.005
lin_svc = svm.SVC(kernel = 'linear', C = C_value).fit(N, m)
scores = cross_val_score(lin_svc, N, m, cv = 5, scoring = 'accuracy')
print("Trivial dataset linear kernel Accuracy for C = ", C_value)
print(scores.mean())

#polynomial kernel with degree and C values tuning:
C_value = 15
degree_value = 3
poly_svc = svm.SVC(kernel='poly', degree = degree_value, C = C_value).fit(N, m)
scores = cross_val_score(poly_svc, N, m, cv = 5, scoring = 'accuracy') 
print("Trivial dataset Polynomial kernel Accuracy for degree =", degree_value, "and C =", C_value)
print(scores.mean())

#Gaussian kernel with C and gamma values tuning:
C_value = 1
gamma_value = 0.1
rbf_svc = svm.SVC(kernel='rbf', gamma = gamma_value , C = C_value).fit(N,m )
scores = cross_val_score(rbf_svc, N, m, cv = 5, scoring = 'accuracy') 
print("Trivial dataset Gaussian kernel Accuracy for C = ", C_value, "and gamma =", gamma_value)
print(scores.mean())











