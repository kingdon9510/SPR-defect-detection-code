#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split


def datacut(filename0)
    # filename0="./data/d2_lab.xlsx"
    data0 = pd.read_excel(filename0)
    data0= np.array(data0)

    # filename1="all_test.xlsx"
    # data1 = pd.read_excel(filename1)
    # data1= np.array(data1)

    # data=np.concatenate((data0,data1),axis=0)
    data=data0

    X=data[:,:-1]
    y=data[:,-1]
    #print(X[1,:])
    #print(y[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
    print (len(X_train), len(X_test))
    #X_test=X_test+40
    y_test=np.resize(y_test,(y_test.shape[0],1))
    y_train=np.resize(y_train,(y_train.shape[0],1))
    lab_train=np.concatenate((X_train,y_train),axis=1)
    lab_test=np.concatenate((X_test,y_test),axis=1)
    # l_tr=pd.DataFrame(lab_train)
    # l_tr.to_excel("./data/data1.xlsx")
    # l_te=pd.DataFrame(lab_test)
    # l_te.to_excel("./data/data2.xlsx")

    # l_tr=pd.DataFrame(lab_train)
    # l_tr.to_excel("./data/d2_lab_train.xlsx",index=None)
    # l_te=pd.DataFrame(lab_test)
    # l_te.to_excel("./data/d2_lab_test.xlsx",index=None)
    return lab_train, lab_test