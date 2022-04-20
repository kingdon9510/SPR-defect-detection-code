import numpy as np
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

from sklearn import preprocessing
import matplotlib.pyplot as plt
#from scalar import scaler
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix,accuracy_score
torch.manual_seed(1234)
# SEED = 123
# BATCH_SIZE = 128
# LEARNING_RATE = 1e-3      # 学习
# EMBEDDING_DIM = 100       # 词向

lr = 0.01
num_epoch =100
BATCH_SIZE =30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=0
bias=0

class BiLSTM_Attention(nn.Module):
    
    def __init__(self,  embedding_dim=256, hidden_dim=100, n_layers=2):
        
        super(BiLSTM_Attention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                           bidirectional=True, dropout=0.5)
       
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.dropout = nn.Dropout(0.5)  
        
    # x: [batch, seq_len, hidden_dim*2]
    # query : [batch, seq_len, hidden_dim * 2]
    # 软注意力机制 (key=value=x)
    def attention_net(self, x, query, mask=None): 
        
        d_k = query.size(-1)     # d_k为query的维�?
       
        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
#         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
#         print("score: ", scores.shape)  # torch.Size([128, 38, 38])
        
        # 对最后一个维�?归一化得�?
        alpha_n = F.softmax(scores, dim=-1) 
#         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)
        
        return context, alpha_n
    
    def forward(self, x):
        # [seq_len, batch, embedding_dim]
        #embedding = self.dropout(self.embedding(x)) 
        
        # output:[seq_len, batch, hidden_dim*2]
        # hidden/cell:[n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(x)
        #print(output.shape)
        #xxx=output[:,-1,-1]
        #print(xxx.shape)
        #output = output.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]
        
        query = self.dropout(output)
        # 加入attention机制
        attn_output, alpha_n = self.attention_net(output, query)
        
        #logit = self.fc(attn_output)
        
        #return logit
        return attn_output

h1l = 200
h2l = 50
h3l = 800
h4l = 400
h5l = 200
h6l = 100
h7l = 16
class ACG(nn.Module):
    def __init__(self):
        super(ACG, self).__init__()
        self.fc1 = nn.Linear(2*300,h1l)
        #self.fc2= nn.Linear(h1l, h2l)
        '''
        self.fc3 = nn.Linear(h2l, h3l)
        self.fc4 = nn.Linear(h3l, h4l)
        self.fc5 = nn.Linear(h4l, h5l)
        self.fc6 = nn.Linear(h5l, h6l)
        self.fc7 = nn.Linear(h6l, h7l)
        '''
        self.fc8 = nn.Linear(h1l, 2)
        self.bn1 = nn.BatchNorm1d(h1l)
        self.bn2 = nn.BatchNorm1d(h2l)
        
        self.bn3 = nn.BatchNorm1d(h3l)
        self.bn4 = nn.BatchNorm1d(h4l)
        self.bn5 = nn.BatchNorm1d(h5l)
        self.bn6 = nn.BatchNorm1d(h6l)
      
        self.drop = nn.Dropout(0.5)

             
 
    def forward(self, x):        
        x = F.relu(self.fc1(x))
        #x = F.relu(self.bn1(x))
        #x = F.relu(self.fc2(x))
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

# 设置device



# 为CPU设置随机种子

def confu(label,pre):
    if len(label)!=len(pre):
        print("wrong")
        exit()
    con=np.zeros((2,2))
    for i in range(len(pre)):
        con[label[i]][pre[i]]+=1
    return con


# Load
cuda = torch.cuda.is_available()
print("CUDA: {}".format(cuda))




#data preprocessing
filename="./data/d1_lab.csv"
#data_train = pd.read_excel(filename)
# data_train = pd.read_csv(filename)
# data_train= np.array(data_train)
data_train,data_test=datacut(filename)

'''
list1=[]
for i in range(data.shape[0]):
    list2=[]
    for t in range(255):
        list2.append(data[i][t+1]-data[i][t])
    list2.append(0)    
    list1.append(list2)
np1=np.array(list1)
data=np.concatenate((np1,data[:,256:]),axis=1)
'''
    
    
    
    
feature=data_train[:,0:256]+bias
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

#xtrain =feature_1

xtrain=torch.from_numpy(xtrain.astype(float))
#=xtrain.view(len(xtrain),1,256)
ytrain = torch.from_numpy(ytrain.astype(float))



#put the data into dataloder
start = time.clock()
train_dataset = Data.TensorDataset(xtrain, ytrain)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2)


list1=[]
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
feature_test=data_test[:,0:256]+bias
#type_t=con(x,data_test)
type_t=data_test[:,256:-1]
type_t1=preprocessing.StandardScaler().fit(type_p).transform(type_t)#normally
feature_test1= preprocessing.StandardScaler().fit(feature).transform(feature_test)#normally
#type_t1=con(x,type_t1)

xtrain_test=np.concatenate((feature_test1,type_t1),axis=1)

ytrain_test=data_test[:,-1]
#print(data[0,256:260])


#xtrain_test=feature_test1
xtrain_test=torch.from_numpy(xtrain_test.astype(float))
#xtrain_test=xtrain_test.view(len(xtrain_test),1,256)
ytrain_test = torch.from_numpy(ytrain_test.astype(float))



#put the data into dataloder

test_dataset = Data.TensorDataset(xtrain_test, ytrain_test)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)



"""
SECTION 2 : Build and Train Model

Multilayer perceptron model, with two hidden layer.
input layer : 248 neuron, represents the feature of flow statics
hidden layer : 300/200 neuron, activation using ReLU
output layer : 12 neuron, represents the class of flows

optimizer = ADAM with 50 batch-size
loss function = categorical cross entropy
learning rate = 0.001
epoch = 500
"""



# Hyperparameters




# Build model

net_acg = BiLSTM_Attention()
if (device):
    net_acg=net_acg.cuda()
fc = nn.Linear(100 * 2+26, 2)
if device:
    fc=fc.cuda()


# Choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(net_acg.parameters(), lr=lr, betas=(0.9, 0.99))
#optimizer = torch.optim.SGD(net_acg.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.SGD(params=chain(net_acg.parameters(),fc.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)

loss_history = []

# Train
net_acg.train()
#fc = ACG()
# fc = nn.Linear(200 * 2, 2)
# if device:
#     fc=fc.cuda()

    
for epoch in range(num_epoch):
    # pre_test=[]
    # label_test=[]
    # pre_train=[]
    # label_train=[]
    for step, (xtrain, ytrain) in enumerate(train_loader):
        if (device):
            xtrain=xtrain.cuda()
            ytrain=ytrain.cuda()
        xtrain = Variable(xtrain.float())
        ytrain = Variable(ytrain.long())
        x=xtrain[:,:256]
        x=x.view(len(x),1,256)
        context=xtrain[:,256:]
        x_out = net_acg(x)
        x_out= torch.cat((x_out,context),1)
        output = fc(x_out)
        #output = net_acg(xtrain)
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
        if (device):
            xtest=xtest.cuda()
            ytest=ytest.cuda()
        xtest = Variable(xtest.float())
        ytest = Variable(ytest.long())
        x=xtest[:,:256]
        x=x.view(len(x),1,256)
        context=xtest[:,256:]
        x_out= net_acg(x)
        x_out= torch.cat((x_out,context),1)
        output = fc(x_out)
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
    print("---------------------------------------------------------------------")
    print(confu(label_test,pre_test))
torch.save(net_acg, 'att_modelss2.pkl')