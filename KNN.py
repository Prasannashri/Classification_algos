import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing Dataset
 dataset = pd.read_csv('social_network_ads.csv')
 iv  = dataset.iloc[:,[2,3]].values #independent variables
 dv = dataset.iloc[:,4].values #dependent variable
#splitting the dataset into training and testing datasets
from sklearn.model_selection import train_test_split
iv_train,iv_test,dv_train,dv_test = train_test_split(iv,dv,test_size=0.2,random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
iv_train = sc_X.fit_transform(iv_train)
iv_test = sc_X.transform(iv_test)
#fitting model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric ='minkowski',p=2)
classifier.fit(iv_train,dv_train)
#prediction
y_pred = classifier.predict(iv_test)
#Identifying correct results using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dv_test,y_pred)
