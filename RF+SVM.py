# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
from time import time

import numpy as np
import matplotlib.pyplot as plt

import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def loadDS():
    """
    Helper function to combine all files into a single numpy array
    Output: tuple
    [0] - 2-D numpy array with images stored in row-major order
    [1] - Encoded labels
    """
    data = []
    labels = []
    for i in range(1,6):
        batch = unpickle('dataset/data_batch_{}'.format(i))
        data.append(batch[b'data'])
        labels.append(batch[b'labels'])
    return np.concatenate(data), np.concatenate(labels)

def confuse(true, pred, classes):
    cm = confusion_matrix(true, pred)
    sns.heatmap(cm.T, square=True, annot=True,
                xticklabels = classes, yticklabels = classes,
                fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')

#Loading Training Data
data, labels = loadDS()
#Loading Testing Data
test = unpickle('dataset/test_batch')
dataT = test[b'data']
labelsT = np.asarray(test[b'labels'], dtype=int)
#Loading Class Names
classes = unpickle('dataset/batches.meta')[b'label_names']
classes = [l.decode('UTF-8') for l in classes]

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

#Converting RGB-images to Y-channel data and scaling them
data = np.reshape(data, (-1,3,32*32))
data = data[:,0,:]*.299 + data[:,1,:]*.587 + data[:,2,:]*.114
data = data.astype(np.float32, copy=False)
data = scaler.fit_transform(data)
            
dataT = np.reshape(dataT, (-1,3,32*32))
dataT = dataT[:,0,:]*.299 + dataT[:,1,:]*.587 + dataT[:,2,:]*.114
dataT = dataT.astype(np.float32, copy=False)
dataT = scaler.transform(dataT)

#Principal Component Analysis
pca = PCA(n_components=35, svd_solver = 'randomized', whiten = True)
print ('Fitting PCA')
data = pca.fit_transform(data, labels)
dataT = pca.transform(dataT)

#Fitting ML models and reporting accuracy
print ('Fitting Boost')
clf = HistGradientBoostingClassifier(max_iter = 5000)
t1 = time()
clf.fit(data,labels)
train_time_boost = time() - t1
pred_GBT = clf.predict(dataT)
acc_GBT = accuracy_score(labelsT, pred_GBT)*100
print (f'Training time: {train_time_boost}s')
print (f'Accuracy: {acc_GBT}%')
confuse(labelsT, pred_GBT, classes)

print ('Fitting RF')
clf = RandomForestClassifier(n_estimators = 1000,
                              max_samples = 0.8,
                              oob_score=True, n_jobs=-1)
t1 = time()
clf.fit(data,labels)
train_time_rf = time() - t1
pred_RF = clf.predict(dataT)
acc_RF = accuracy_score(labelsT, pred_RF)
print (f'Training time: {train_time_rf}s')
print (f'Accuracy: {acc_RF}%')
confuse(labelsT, pred_RF, classes)

print ('Training SVM')
clf = SVC(gamma=0.1, max_iter=2500)
t1 = time()
clf.fit(data, labels)
train_time_svm = time() - t1
pred_SVM = clf.predict(dataT)
acc_SVM = accuracy_score(labelsT, pred_SVM)*100
print (f'Training time: {train_time_svm}s')
print (f'Accuracy: {acc_SVM} %')
confuse(labelsT, pred_SVM, classes)
