from __future__ import division
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import GWO as gwo
import numpy as np # linear algebra
import pandas as pd
from sklearn.preprocessing import StandardScaler




def selector(popSize,Iter):
    
    data = shuffle(np.array(pd.read_csv("datasets/Dataset.csv",header=1)))
    
    #print(data.info())
    
    #print(data)    
    extracted_dataset= []
    target = []
    
    for row in data:
        extracted_dataset.append(row[0:-2])
        target.append(row[-1])
    
    scaler = StandardScaler()
    new_data = scaler.fit_transform(extracted_dataset)
    
    print(new_data[0])
    X_train, X_test, Y_train, Y_test= train_test_split(new_data,target,test_size=0.3)
    
    dim=len(X_train[0])
    
    x=gwo.GWO(dim,popSize,Iter,X_train,Y_train,X_test,Y_test)

    return x
    
#####################################################################    
