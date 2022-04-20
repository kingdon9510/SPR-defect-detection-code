
"""
SECTION 1 : Load and setup data for training

"""
import pandas as pd
import numpy as np
import time
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from class_251 import ACG
from class_1dcnn import ACG
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from scalar import scaler
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix,accuracy_score
torch.manual_seed(1234)
from data_cut import *

import torch.autograd as autograd

h1l = 1000
h2l = 1100
h3l = 600
h4l = 800
h5l = 600
h6l = 400
h7l = 200
class ACG(nn.Module):
    def __init__(self):
        super(ACG, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 25, 5, stride=3, padding=2),  # b, 16, 10, 10
            nn.BatchNorm1d(25),
            nn.ReLU(True)
            )
        self.max1=nn.MaxPool1d(2, stride=2,padding=1)
        self.conv2=nn.Sequential(nn.Conv1d(25, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.BatchNorm1d(8),
            nn.ReLU(True)
            )
        self.max2=nn.MaxPool1d(2, stride=2)  # b, 8, 2, 2
        

        self.fc1 = nn.Linear(168,h1l)
        self.fc2= nn.Linear(h1l, h2l)
        
        self.fc3 = nn.Linear(h2l, h3l)
        self.fc4 = nn.Linear(h3l, h4l)
        self.fc5 = nn.Linear(h4l, h5l)
        self.fc6 = nn.Linear(h5l, h6l)
        self.fc7 = nn.Linear(h6l, h7l)
        self.fc8 = nn.Linear(h2l, 2)
        self.bn1 = nn.BatchNorm1d(h1l)
        self.bn2 = nn.BatchNorm1d(h2l)
        self.bn3 = nn.BatchNorm1d(h3l)
        self.bn4 = nn.BatchNorm1d(h4l)
        self.bn5 = nn.BatchNorm1d(h5l)
        self.bn6 = nn.BatchNorm1d(h6l)
        self.drop = nn.Dropout(0.5)

             
 
    def forward(self,x):
        x = self.conv1(x)
        x=self.max1(x)
        x=self.conv2(x)
        x1=self.max2(x)
        print(x1.shape)
        
        #x=x1.reshape(x1.shape[0],-1)
        #x=torch.cat((x,context),1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.bn1(x))
        x = F.relu(self.fc2(x))
        '''
        #x = F.relu(self.bn2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.bn3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.drop(x))
        #x = F.relu(self.bn4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.bn5(x))
        x = F.relu(self.fc6(x))
        #x = F.relu(self.bn6(x))
        x = F.relu(self.fc7(x))
        '''
        #x = F.relu(self.fc8(x))
        x2 = self.fc8(x)
        return x2


BATCH_SIZE = 50
lr = 0.01
num_epoch = 40

def confu(label,pre):
    if len(label)!=len(pre):
        print("wrong")
        exit()
    conf=np.zeros((2,2))
    for i in range(len(pre)):
        conf[label[i]][pre[i]]+=1
    return conf

def con(x,data):
    cc=[]
    for i in range(len(x)):
        cc.append(np.resize(data[:,x[i]],(data[:,x[i]].shape[0],1)))
    tt = cc[0]
    for i in range(len(x)-1):
        tt=np.concatenate((tt,cc[i+1]),axis=1)
    return tt




# Load
device = torch.cuda.is_available()
print("CUDA: {}".format(cuda))



#data preprocessing
filename="./data/d1_lab.csv"
#data_train = pd.read_excel(filename)
# data_train = pd.read_csv(filename)
# data_train= np.array(data_train)
data_train,data_test=datacut(filename)
'''
for i in range(data_test.shape[0]):
    list2=[]
    for t in range(255):
        list2.append(data_test[i][t+1]-data_test[i][t])
    list2.append(0)    
    list1.append(list2)
np1=np.array(list1)
data_test=np.concatenate((np1,data_test[:,256:]),axis=1)
'''
x=[0, 1, 4, 5, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25]
feature=data_train[:,0:256]
type_p=data_train[:,256:-1]
#type_p=con(x,data_train)
type_p1=preprocessing.StandardScaler().fit_transform(type_p)#normally
feature_1= preprocessing.StandardScaler().fit_transform(feature)#normally
#type_p1=con(x,type_p1)
#context=np.concatenate((context_123,context_4,context_5),axis=1)
xtrain=np.concatenate((feature_1,type_p1),axis=1)
#xtrain=np.concatenate((feature,context),axis=1)
ytrain=data_train[:,-1]
#print(data[0,256:260])



xtrain=torch.from_numpy(xtrain.astype(float))
#xtrain=xtrain.view(len(xtrain),1,784)
ytrain = torch.from_numpy(ytrain.astype(float))



#put the data into dataloder
start = time.clock()
train_dataset = Data.TensorDataset(xtrain, ytrain)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2)

# #filename="lab_test.csv"
# filename="site.xlsx"
# #data_test = pd.read_excel(filename)
# #data_test =pd.read_csv(filename)
# data_test =pd.read_excel(filename)
# data_test= np.array(data_test)
'''
list1=[]

for i in range(data_test.shape[0]):
    list2=[]
    for t in range(255):
        list2.append(data_test[i][t+1]-data_test[i][t])
    list2.append(0)    
    list1.append(list2)
np1=np.array(list1)
data_test=np.concatenate((np1,data_test[:,256:]),axis=1)
'''
feature_test=data_test[:,0:256]
#type_t=con(x,data_test)
type_t=data_test[:,256:-1]
type_t1=preprocessing.StandardScaler().fit(type_p).transform(type_t)#normally
feature_test1= preprocessing.StandardScaler().fit(feature).transform(feature_test)#normally
type_t1=con(x,type_t1)

xtrain_test=np.concatenate((feature_test1,type_t1),axis=1)
ytrain_test=data_test[:,-1]
#print(data[0,256:260])



xtrain_test=torch.from_numpy(xtrain_test.astype(float))
#xtrain_test=xtrain.view(len(xtrain_test),1,784)
ytrain_test = torch.from_numpy(ytrain_test.astype(float))



#put the data into dataloder

test_dataset = Data.TensorDataset(xtrain_test, ytrain_test)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)

net_acg=ACG()
if device:
    net_acg=net_acg.cuda()
net_acg.train()

# Choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(net_acg.parameters(), lr=lr, betas=(0.9, 0.99))
optimizer = torch.optim.SGD(net_acg.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
'''
from itertools import chain
optim.Adam(params=chain(backbone_net.parameters(),
                                                linear_rot_net.parameters(),
                                                linear_classify_net.parameters()),
'''
loss_history = []

# Train


for epoch in range(num_epoch):
    # pre_test=[]
    # label_test=[]
    # pre_train=[]
    # label_train=[]
    for step, (xtrain, ytrain) in enumerate(train_loader):
        if device:
            xtrain=xtrain.cuda()
            ytrain=ytrain.cuda()
        xtrain = Variable(xtrain.float())
        ytrain = Variable(ytrain.long())
        #x=xtrain[:,0:NF]
        #x=x.view(len(x),1,NF)
        #context=xtrain[:,256:]
        #output = net_acg(x,context)
        output = net_acg(xtrain)
        loss = criterion(output,ytrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # _, predicted = torch.max(output.data, 1)
        # predicted=predicted.detach()
        # predicted=predicted.cpu().int()
        # pre_train.extend(list(np.array(predicted))) 
        # label_train.extend(list(np.array(ytrain)))    
    if epoch % 10== 0 and epoch != 0:
        lr = lr * 0.8
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    previous_loss = loss.data
    loss_history.append(loss.data)    
    print("epoch:",epoch,loss.data)  
    pre_test=[]
    label_test=[]
    pre_train=[]
    label_train=[]
    for step, (xtest, ytest) in enumerate(test_loader):
        if device:
            xtest=xtest.cuda()
            ytest=ytest.cuda()       
        xtest = Variable(xtest.float())
        ytest = Variable(ytest.long())
        #x=xtrain[:,0:NF]
        #x=x.view(len(x),1,NF)
        #context=xtrain[:,256:]
        output = net_acg(xtest)
        #output = net_acg(x)
        #soft=F.softmax(output,dim=1)
        #soft=soft.detach()
        #soft=soft.cpu()        
        soft, predicted = torch.max(F.softmax(output.data,dim=1), 1)
        soft=soft.detach()
        soft=soft.cpu()         
        predicted=predicted.detach()
        predicted=predicted.cpu().int()
        pre_test.extend(list(np.array(predicted))) 
        label_test.extend(list(np.array(ytest.cpu())))  

    accuracy=accuracy_score(label_test, pre_test)
    precision_test=precision_score(label_test,pre_test)
    recall_test=recall_score(label_test,pre_test)
    f_score_test=f1_score(label_test,pre_test)
    #print('conf_mat2:\n',conf_mat2) 
    #print('accuracy:%4f %%'%(100*float(correct)/len(label)))
    print('accuracy:',accuracy)
    print('fsocre:',f_score_test)
    print('recall_score',recall_test)
    print('precision_score',precision_test)
#torch.save(net_acg, 'att3_model.pkl')
