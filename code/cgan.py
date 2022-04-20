import numpy as np

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F 
import pandas as pd
import numpy as np
import math
import time
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
#from class_251 import ACG
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from scalar import scaler
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix,accuracy_score

import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from data_cut import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,MaxAbsScaler,RobustScaler
from sklearn import metrics
#from xgboost.sklearn import XGBClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from numpy.random import seed
seed(1)  
torch.manual_seed(1234)
# SEED = 123
# BATCH_SIZE = 128
# LEARNING_RATE = 1e-3      # 学习�?
# EMBEDDING_DIM = 100       # 词向量维�?

lr = 0.01
num_epoch =100
BATCH_SIZE =10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=0
bias=0
z_dimension=100


def con(x,data):
    cc=[]
    for i in range(len(x)):
        cc.append(np.resize(data[:,x[i]],(data[:,x[i]].shape[0],1)))
    tt = cc[0]
    for i in range(len(x)-1):
        tt=np.concatenate((tt,cc[i+1]),axis=1)
    return tt
def confu(label,pre):
    if len(label)!=len(pre):
        print("wrong")
        exit()
    conf1=np.zeros((2,2))
    for i in range(len(pre)):
        conf1[int(label[i])][int(pre[i])]+=1
    return conf1
def FPR(confu):
    return float(confu[0][1]/(confu[0][0]+confu[0][1]))



class JDA:
    def __init__(self, T=40):

        self.T = T

    def fit_predict(self, Xs, Ys, Xt, Yt,train_t,test_t):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''

        clf0 = MLPClassifier(hidden_layer_sizes = (15),max_iter =150)
        clf1 =sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
        clf2 = DecisionTreeClassifier(max_depth=10)
        clf3 = RandomForestClassifier()
        clf4 = svm.SVC(probability=True)
        clf5 = GaussianNB()
        clfs =[clf0,clf1,clf2,clf3]

        for t in range(T):
            # xst=np.hstack((Xs,train_t))
            # xtt=np.hstack((Xt,test_t))
            xst=Xs
            xtt=Xt

            acc=[]
            tpr=[]
            fpr=[]
            pr=[]
            f1=[]
            print("---------------Epoch"+str(t)+"----------------")
            print("ACC  TPR/RC  FPR   PR   F1")
            for i in range (len(clfs)):
                clfs[i].fit(xst ,Ys.ravel())
                Y_tar = clfs[i].predict(xtt)
                #time2 = time.time()
                #print('time:',time2-time1)
                #print('average time:',(time2-time0)/(t+1))		
                #clf0 = MLPClassifier(hidden_layer_sizes = (15),max_iter = 50)
                #clf0=sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)

                #clf0 = DecisionTreeClassifier(max_depth=10)
                #clf0 = RandomForestClassifier()
                #clf0 = svm.SVC(probability=True)
                #clft[i].fit(Xt_new, Y_tar_pseudo.ravel())
                #Y_tar = clft[i].predict(Xt_new)
                #acc = sklearn.metrics.accuracy_score(Yt, Y_tar)
                acc.append(metrics.accuracy_score(Yt, Y_tar))
                tpr.append(metrics.recall_score(Yt, Y_tar))
                fpr.append(FPR(confu(Yt, Y_tar)))
                pr.append(metrics.precision_score(Yt, Y_tar))
                f1.append(metrics.f1_score(Yt, Y_tar))
                
                print('{:.3f}   {:.3f}   {:.3f}    {:.3f}   {:.3f}'.format(acc[-1],tpr[-1],fpr[-1],pr[-1],f1[-1]))
                #list_acc.append(acc)
            #print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, acc))
            #print(metrics.classification_report(Yt, Y_tar))
            #trainset= np.append(Xs_new,Ys.reshape(-1,1),axis=1)
            #testset= np.append(Xt_new,Yt.reshape(-1,1),axis=1)
            #np.savetxt('/home/ustc-1/pu/final/MLP/train/'+str(t+1)+'.csv',trainset,delimiter=',')
            #np.savetxt('/home/ustc-1/pu/final/MLP/test/'+str(t+1)+'.csv',testset,delimiter=',')
        #print(list_acc)
        return 1


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(292,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,2)
            #nn.Sigmoid()

        )
    def forward(self, x):
        x=self.dis(x)
        return x
 
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(z_dimension+36,200),
            nn.ReLU(True),
            nn.Linear(200,200),
            nn.ReLU(True),
            nn.Linear(200,256)
        )
 
    def forward(self, x):
        x=self.gen(x)
        return x

cuda = torch.cuda.is_available()
print("CUDA: {}".format(cuda))



# #data preprocessing
# filename="./data/d2_lab_train.xlsx"
# #filename="./data/d1_lab.xlsx"
# data_train = pd.read_excel(filename)
# data_train= np.array(data_train)   
filename="./data/d1_lab.csv"

data_train,data_test=datacut(filename)


feature=data_train[:,0:256]+bias
type_p=data_train[:,256:-1]
#type_p=con(x,data_train)
type_p1=StandardScaler().fit_transform(type_p)#normally
feature_1= StandardScaler().fit_transform(feature)#normally
#type_p1=con(x,type_p1)
#context=np.concatenate((context_123,context_4,context_5),axis=1)
xtrain=np.concatenate((feature_1,type_p1),axis=1)
#xtrain=np.concatenate((feature,context),axis=1)
ytrain=data_train[:,-1]

data_train_1=[]

# select the defected samples
for i in range(data_train.shape[0]):
    if data_train[i][-1]==1:
        data_train_1.append(np.hstack((xtrain[i,:],[1])))
data_train_1=np.array(data_train_1)

#print(data[0,256:260])

#xtrain =feature_1

xtrain=torch.from_numpy(xtrain.astype(float))
#=xtrain.view(len(xtrain),1,256)
ytrain = torch.from_numpy(ytrain.astype(float))
#############################################################
ytrain1=ytrain.view(ytrain.shape[0],1)
train_data=torch.cat((xtrain,ytrain1),1)

#############################################################

data_test=data_train_1
# feature_test=data_test[:,0:256]+bias
# #type_t=con(x,data_test)
# type_t=data_test[:,256:-1]
# type_t1=preprocessing.StandardScaler().fit(type_p).transform(type_t)#normally
# feature_test1= preprocessing.StandardScaler().fit(feature).transform(feature_test)#normally
# #type_t1=con(x,type_t1)

# xtrain_test=np.concatenate((feature_test1,type_t1),axis=1)
xtrain_test=data_train_1[:,:-1]
print(xtrain_test.shape)

ytrain_test=data_train_1[:,-1]

#print(data[0,256:260])


#xtrain_test=feature_test1
xtrain_test=torch.from_numpy(xtrain_test.astype(float))
#xtrain_test=xtrain_test.view(len(xtrain_test),1,256)
ytrain_test = torch.from_numpy(ytrain_test.astype(float))



#put the data into dataloder

test_dataset = Data.TensorDataset(xtrain_test, ytrain_test)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)



#put the data into dataloder
start = time.clock()
train_dataset = Data.TensorDataset(xtrain, ytrain)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2)

D=discriminator()
G=generator()
if torch.cuda.is_available():
    D=D.cuda()
    G=G.cuda()

criterion = nn.CrossEntropyLoss()
g_optimizer = torch.optim.SGD(G.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
d_optimizer = torch.optim.SGD(D.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

extra_img=[]
flag1=0
for epoch in range(num_epoch):
    for i,(img, y) in enumerate(train_loader):
        num_img=img.size(0)

        # img = img.view(num_img, -1)  
        real_img = Variable(img[:,:256].float()).cuda() 
        t=Variable(img[:,256:].float()).cuda()
        y=y.view(y.shape[0],1)
        y=Variable(y.float()).cuda()
        t=torch.cat((t,y,y,y,y,y,y,y,y,y,y),1)
        real_label = Variable(torch.ones(num_img).long()).cuda()  
        fake_label = Variable(torch.zeros(num_img).long()).cuda()  
        

        real_out = D(torch.cat((real_img,t),1)) 
        d_loss_real = criterion(real_out, real_label)
        #real_scores = real_out  -------------
 

        z = Variable(torch.randn(num_img, z_dimension).float()).cuda() 
        fake_img = G(torch.cat((z,t),1)) 
        fake_out = D(torch.cat((fake_img,t),1))  
        d_loss_fake = criterion(fake_out, fake_label) 
        #fake_scores = fake_out  -------------
 

        d_loss = d_loss_real + d_loss_fake
        #d_loss = d_loss_real
        d_optimizer.zero_grad()  
        d_loss.backward()  
        if i % 5==0:
            z = Variable(torch.randn(num_img, z_dimension)).cuda()  
            fake_img = G(torch.cat((z,t),1)) 
            #print((fake_img.data.cpu())))
            output = D(torch.cat((fake_img,t),1))  
            g_loss = criterion(output, real_label)


            g_optimizer.zero_grad()  
            g_loss.backward() 
            g_optimizer.step() 
    print("epoch:",epoch)
    print("d_loss:",d_loss.data)
    print("d_loss_real:",d_loss_real.data)
    print("g_loss:",g_loss.data)
    print("------------------------------")
    if epoch>89:
        for step, (xtest, ytest) in enumerate(test_loader):
            num=xtest.size(0)
            if (device):
                xtest=xtest.cuda()
                ytest=ytest.cuda()
            xtest = Variable(xtest.float())
            ytest = Variable(ytest.float())
            #print(ytest)
            ytest=ytest.view(ytest.shape[0],1)
            x=xtest[:,:256]
            context=xtest[:,256:]
            t=torch.cat((context,ytest,ytest,ytest,ytest,ytest,ytest,ytest,ytest,ytest,ytest),1)
            z = Variable(torch.randn(num, z_dimension).float()).cuda()
            fake_img = G(torch.cat((z,t),1))
            fake_img = torch.cat((fake_img.data,context,ytest),1)
            #print(fake_img.shape)
            #print(np.array(fake_img.cpu()))
            train_data=np.vstack((np.array(train_data),np.array(fake_img.cpu())))

train_data_all=train_data





T=256
#data_S = pd.read_csv("lab_train.csv")
#data_S = pd.read_excel("./data/d1_lab.xlsx")
#print(data_S)
data_S = np.array(train_data_all)
#data_S=train_data_all
train_t = data_S[:,T:-1]
train_S = data_S[:,:T]#瀹炰�?
#train_S= RobustScaler().fit_transform(train_S)
#train_S= Normalizer().fit_transform(train_S)
#train_S= MinMaxScaler().fit_transform(train_S)
#train_S= StandardScaler().fit_transform(train_S0)
#train_t=StandardScaler().fit_transform(train_t0)
label_S = data_S[:,-1]#绫诲埆鏍囩


#data_T = pd.read_csv("testset.csv")
data_T=data_test

test = data_T[:,:T]#瀹炰�?
#test= RobustScaler().fit_transform(test)
#test= Normalizer().fit_transform(test)
#test= MinMaxScaler().fit_transform(test)
test= StandardScaler().fit(feature).transform(test)
label_test = data_T[:,-1]#绫诲埆鏍囩
test_t = data_T[:,T:-1]
test_t=StandardScaler().fit(type_p).transform(test_t)
print(train_S.shape,train_t.shape,label_S.shape,test.shape,test_t.shape, label_test.shape)

#    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
#clf = GaussianNB()
#clf = MLPClassifier(hidden_layer_sizes = (15),max_iter = 300)

#clf = RandomForestClassifier()
#clf = svm.SVC(probability=True)
#clf = DecisionTreeClassifier(max_depth=10)
#    clf.fit(train_S,label_S )

#    clf.fit(train_S,label_S )
#    y_pred0 = clf.predict(train_S)
#    acc0 = sklearn.metrics.accuracy_score(label_S, y_pred0)
#    print(acc0)
#    print(metrics.classification_report(label_S, y_pred0))
#    print('--------------------------source-only-------------------------------')
#    y_pred = clf.predict(test)
#    acc1 = sklearn.metrics.accuracy_score(label_test, y_pred)
#    print(acc1)
#    print(metrics.classification_report(label_test, y_pred))
#    print('--------------------------JDA-Target-------------------------------')


jda = JDA()
acc1 = jda.fit_predict(train_S,label_S, test, label_test,train_t,test_t)
#print(acc)
#print(metrics.classification_report(label_test, ypre))

    










