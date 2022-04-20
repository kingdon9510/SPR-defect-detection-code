#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from minepy import MINE
from scipy import stats
import random



def printmic(huxinxi):
    list_l=[]
    print("-------------MIC------------")
    for i in range(26):

        if float(huxinxi[i]) >= 0.0001:
            list_l.append(i)
        print(huxinxi[i],i)
    print(list_l,len(list_l))
    print(np.sort(huxinxi))
filename='./data/d2_lab.xlsx'

m = MINE()

data = pd.read_excel(filename)
data= np.array(data)
y=data[1:,-1]

all_person_p=[]
all_person_r=[]
huxinxi=[]
flag2=1
flag3=1
for i in np.arange(1,len(data),1):
    if data[i][10+256]==0 and data[i][14+256]==0 and data[i][18+256]==0:
        if flag2==1:
            data_2l=data[i,:]
            flag2=0
        else:
            data_2l=np.vstack((data_2l,data[i,:]))
    else:
        if flag3==1:
            data_3l=data[i,:]
            flag3=0
        else:
            data_3l=np.vstack((data_3l,data[i,:]))
huxinxi_2l=[]
huxinxi_3l=[]



# for i in range(26):
#     m.compute_score(data_2l[:,i+256],data_2l[:,-1])
#     huxinxi_2l.append(format(m.mic(),'.4f'))
# printmic(huxinxi_2l)
# for i in range(26):
#     m.compute_score(data_3l[:,i+256],data_3l[:,-1])
#     huxinxi_3l.append(format(m.mic(),'.4f'))
# printmic(huxinxi_3l)
# listaver_2=[]

# for i in range(data_2l.shape(0)):
#     listaver_2.append()

#aver=np.mean(data_2l[:,:256],1)
aver=np.var(data_2l[:,:256],1)
m.compute_score(aver,data_2l[:,-1])
print(format(m.mic(),'.4f'))
sum_thick=np.sum(data_2l[:,272:18+256],1)

m.compute_score(sum_thick,data_2l[:,-1])
print(format(m.mic(),'.4f'))
total_x=data_2l[:,25+256]*256
m.compute_score(total_x,data_2l[:,-1])
print(format(m.mic(),'.4f'))


#aver=np.mean(data_3l[:,:256],1)
aver=np.var(data_3l[:,:256],1)
m.compute_score(aver,data_3l[:,-1])
print(format(m.mic(),'.4f'))
sum_thick=np.sum(data_3l[:,272:19+256],1)
m.compute_score(sum_thick,data_3l[:,-1])
print(format(m.mic(),'.4f'))
total_x=data_3l[:,25+256]*256
m.compute_score(total_x,data_3l[:,-1])
print(format(m.mic(),'.4f'))


# print("----------person_r----------")
# for i in range(13):
#     print(all_person_r[i])
# print("----------person_p----------")    
# for i in range(13):    
#     print(all_person_p[i])

