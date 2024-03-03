import pickle,pdb

import torch
import torch.utils.data as data
import numpy as np
import torch.nn.functional as F
from torch import optim,nn
import matplotlib.pyplot as plt

with open('../pkl/stoi.pkl','rb') as f:
    sztoi=pickle.load(f)
with open('../pkl/vectors.pkl','rb') as f:
    iztov=pickle.load(f)
with open('../pkl/train_pru_txt_list.pkl','rb') as f:
    train_pru_txt_list=pickle.load(f)
with open('../pkl/test_pru_txt_list.pkl','rb') as f:
    test_pru_txt_list=pickle.load(f)
with open('../pkl/score.pkl','rb') as f:
    score=pickle.load(f)
print(f'score\'s lenth={len(score[0])}')

# score的预处理——分成2类

sample_num=1000
train_score_label=[[1,0]]*12500+[[0,1]]*12500
test_score_label=[[1,0]]*12500+[[0,1]]*12500

# pdb.set_trace()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class trainDataset(data.Dataset):
    def __init__(self):
        self.x=train_pru_txt_list[:sample_num]+train_pru_txt_list[12500:12500+sample_num]
        self.y=torch.tensor(train_score_label[:sample_num]+train_score_label[12500:12500+sample_num]).float()

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)

class testDataset(data.Dataset):
    def __init__(self):
        self.x=test_pru_txt_list[:sample_num]+test_pru_txt_list[12500:12500+sample_num]
        self.y=torch.tensor(test_score_label[:sample_num]+test_score_label[12500:12500+sample_num]).float()

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)


train_data=trainDataset()
test_data=testDataset()
train_loader=data.DataLoader(train_data,shuffle=True,batch_size=64,drop_last=True)
test_loader=data.DataLoader(test_data,shuffle=True,batch_size=64,drop_last=True)

# train_loader=data.DataLoader(train_data,shuffle=True,drop_last=True)
# test_loader=data.DataLoader(test_data,shuffle=True,drop_last=True)
print(len(train_data))
print(len(test_data))


import torch.nn as nn
import torch

class ALSTM_SIN(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,fc_hidden_size,output_size):
        super(ALSTM_SIN,self).__init__()

        self.input_size=input_size
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.output_size=output_size

        self.w_ig=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,input_size)))
        self.w_ii=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,input_size)))
        self.w_if=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,input_size)))
        self.w_io=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,input_size)))

        self.w_hg=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))
        self.w_hi=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))
        self.w_hf=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))
        self.w_ho=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))

        self.b_g=nn.Parameter(torch.zeros(batch_size, hidden_size,1))
        self.b_i=nn.Parameter(torch.zeros(batch_size, hidden_size,1))
        self.b_f=nn.Parameter(torch.zeros(batch_size, hidden_size,1))
        self.b_o=nn.Parameter(torch.zeros(batch_size, hidden_size,1))

        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.logsoftmax=nn.LogSoftmax(dim=1)

        self.classfier=nn.Sequential(
            nn.Linear(hidden_size,fc_hidden_size),
            nn.Tanh(),
            nn.Linear(fc_hidden_size,output_size)
        )

        self.hidd=self.init_hidd()
        self.cell=self.init_cell()

    def init_hidd(self):
        return torch.zeros(self.batch_size,self.hidden_size,1).to(device)

    def init_cell(self):
        return torch.zeros(self.batch_size,self.hidden_size,1).to(device)

    def forward(self,input):
        # input has size(seq_len,batch_size,input_size)
        # hidden has size(batch_size,hidden_size,1)
        # cell has size(batch_size,hidden_size,1)

        output=torch.zeros(input.size(0),input.size(1),self.output_size)
        for i in range(len(input)):
            x_t=input[i]
            x_t=x_t.unsqueeze(-1)       # x=(batch_size,input_size,1)
            # print(f'device2:{self.w_ig.device,x_t.device}')
            g_t=self.tanh(torch.bmm(self.w_ig,x_t)+torch.bmm(self.w_hg,self.hidd)+self.b_g)      # output (batch_size,hidden_size,1)
            i_t=self.sigmoid(torch.bmm(self.w_ii,x_t)+torch.bmm(self.w_hi,self.hidd)+self.b_i)
            f_t=self.sigmoid(torch.bmm(self.w_if,x_t)+torch.bmm(self.w_hf,self.hidd)+self.b_f)
            o_t=self.sigmoid(torch.bmm(self.w_io,x_t)+torch.bmm(self.w_ho,self.hidd)+self.b_o)
            self.cell=torch.mul(f_t,self.cell)+torch.mul(i_t,g_t)   # output (batch_size,hidden_size,1)
            self.hidden=torch.mul(o_t,self.tanh(self.cell))       
            # output[i]=self.classfier(self.hidden.squeeze(-1))   #(batch_size,output_size)
            # output[i]=self.logsoftmax(output[i])
            if i==len(input)-1:       
                target=self.classfier(self.hidden.squeeze(-1))   #(batch_size,output_size)
                target=self.logsoftmax(target)
        # return output[-1]
        return target

class ALSTM_MUL(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,fc_hidden_size,output_size,n_layer):
        super(ALSTM_MUL,self).__init__()

        self.input_size=input_size
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.output_size=output_size
        self.n_layer=n_layer

        self.w_ig=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,input_size)))
        self.w_ii=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,input_size)))
        self.w_if=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,input_size)))
        self.w_io=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,input_size)))

        self.w_ig2=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))
        self.w_ii2=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))
        self.w_if2=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))
        self.w_io2=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))
        
        self.w_hg=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))
        self.w_hi=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))
        self.w_hf=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))
        self.w_ho=nn.Parameter(nn.init.xavier_normal_(torch.empty(batch_size, hidden_size,hidden_size)))

        self.b_g=nn.Parameter(torch.zeros(batch_size, hidden_size,1))
        self.b_i=nn.Parameter(torch.zeros(batch_size, hidden_size,1))
        self.b_f=nn.Parameter(torch.zeros(batch_size, hidden_size,1))
        self.b_o=nn.Parameter(torch.zeros(batch_size, hidden_size,1))

        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.logsoftmax=nn.LogSoftmax(dim=1)

        self.classfier=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size,fc_hidden_size),
            nn.Tanh(),
            nn.Linear(fc_hidden_size,output_size)
        )

        self.hidd=self.init_hidd()
        self.cell=self.init_cell()

    def init_hidd(self):
        return torch.zeros(self.batch_size,self.hidden_size,1).to(device)

    def init_cell(self):
        return torch.zeros(self.batch_size,self.hidden_size,1).to(device)

    def hidd_forward(self,hidd_vari):
        self.cell=self.init_cell().to(device)
        self.hidd=self.init_hidd().to(device)
        for i in range(len(hidd_vari)):
            x_t=hidd_vari[i]
            x_t=x_t.unsqueeze(-1)       # x=(batch_size,input_size,1)
            # print(f'device2:{self.w_ig.device,x_t.device}')
            g_t=self.tanh(torch.bmm(self.w_ig2,x_t)+torch.bmm(self.w_hg,self.hidd)+self.b_g)      # output (batch_size,hidden_size,1)
            i_t=self.sigmoid(torch.bmm(self.w_ii2,x_t)+torch.bmm(self.w_hi,self.hidd)+self.b_i)
            f_t=self.sigmoid(torch.bmm(self.w_if2,x_t)+torch.bmm(self.w_hf,self.hidd)+self.b_f)
            o_t=self.sigmoid(torch.bmm(self.w_io2,x_t)+torch.bmm(self.w_ho,self.hidd)+self.b_o)
            self.cell=torch.mul(f_t,self.cell)+torch.mul(i_t,g_t)   # output (batch_size,hidden_size,1)
            self.hidd=torch.mul(o_t,self.tanh(self.cell))       
            # print(f'hidd_size={self.hidd.shape}')
            hidd_vari[i]=self.hidd.squeeze(-1)  #(batch_size,output_size)
        return hidd_vari

    def forward(self,input):
        # input has size(seq_len,batch_size,input_size)
        # hidden has size(batch_size,hidden_size,1)
        # cell has size(batch_size,hidden_size,1)

        output=torch.zeros(input.size(0),input.size(1),self.hidden_size).to(device)
        for i in range(len(input)):
            x_t=input[i]
            x_t=x_t.unsqueeze(-1)       # x=(batch_size,input_size,1)
            # print(f'device2:{self.w_ig.device,x_t.device}')
            g_t=self.tanh(torch.bmm(self.w_ig,x_t)+torch.bmm(self.w_hg,self.hidd)+self.b_g)      # output (batch_size,hidden_size,1)
            i_t=self.sigmoid(torch.bmm(self.w_ii,x_t)+torch.bmm(self.w_hi,self.hidd)+self.b_i)
            f_t=self.sigmoid(torch.bmm(self.w_if,x_t)+torch.bmm(self.w_hf,self.hidd)+self.b_f)
            o_t=self.sigmoid(torch.bmm(self.w_io,x_t)+torch.bmm(self.w_ho,self.hidd)+self.b_o)
            self.cell=torch.mul(f_t,self.cell)+torch.mul(i_t,g_t)   # output (batch_size,hidden_size,1)
            self.hidd=torch.mul(o_t,self.tanh(self.cell))       
            # print(f'hidd_size={self.hidd.shape}')
            output[i]=self.hidd.squeeze(-1)  #(batch_size,output_size)
            # if i==len(input)-1:       
            #     target=self.classfier(self.hidden.squeeze(-1))   #(batch_size,output_size)
            #     target=self.logsoftmax(target)

        if(self.n_layer>=3):
            for i in range(self.n_layer-2):
                output=self.hidd_forward(output)

        self.cell=self.init_cell().to(device)
        self.hidd=self.init_hidd().to(device)

        for i in range(len(output)):
            x_t=output[i]
            x_t=x_t.unsqueeze(-1)       # x=(batch_size,input_size,1)
            # print(f'device2:{self.w_ig.device,x_t.device}')
            # print(self.w_ig2.device,x_t.device,self.hidd.device)
            g_t=self.tanh(torch.bmm(self.w_ig2,x_t)+torch.bmm(self.w_hg,self.hidd)+self.b_g)      # output (batch_size,hidden_size,1)
            i_t=self.sigmoid(torch.bmm(self.w_ii2,x_t)+torch.bmm(self.w_hi,self.hidd)+self.b_i)
            f_t=self.sigmoid(torch.bmm(self.w_if2,x_t)+torch.bmm(self.w_hf,self.hidd)+self.b_f)
            o_t=self.sigmoid(torch.bmm(self.w_io2,x_t)+torch.bmm(self.w_ho,self.hidd)+self.b_o)
            self.cell=torch.mul(f_t,self.cell)+torch.mul(i_t,g_t)   # output (batch_size,hidden_size,1)
            self.hidd=torch.mul(o_t,self.tanh(self.cell))       
            # output[i]=self.classfier(self.hidd.squeeze(-1))   #(batch_size,output_size)
            # output[i]=self.logsoftmax(output[i])
            if i==len(input)-1:       
                target=self.classfier(self.hidd.squeeze(-1))   #(batch_size,output_size)
                target=self.logsoftmax(target)
        # return output[-1]
        return target

class LSTM_last_ele(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers,batch_size,output_size):
        super(LSTM_last_ele, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, bidirectional=True, batch_first=False)
        self.layer = nn.Sequential(nn.Linear(2*self.hidden_size, 32), 
                        nn.Tanh(),nn.Linear(32,output_size),
                        nn.Softmax(dim=-1))
        self.hidd=self.init_hidd()
        self.cell=self.init_cell()
        
    def init_hidd(self):
        return torch.zeros(self.n_layers*2,self.batch_size,self.hidden_size).to(device)

    def init_cell(self):
        return torch.zeros(self.n_layers*2,self.batch_size,self.hidden_size).to(device)


    def forward(self,input):
        output, (self.hidd,self.cell) = self.lstm(input, (self.hidd, self.cell))    # 592,64,100->592,64,128
        output = self.layer(output[-1, :, :])           #592,64,128->64,128,只取最后一个元素,64,128->164,10
        return output

n_layer=1
vector_size=100
hidden_size=128
batch_size=64
output_size=2
fc_hidden_size=32
net=LSTM_last_ele(vector_size,hidden_size,n_layer,batch_size,output_size)
# net=ALSTM_SIN(vector_size,hidden_size,batch_size,fc_hidden_size,output_size)
# net=ALSTM_MUL(vector_size,hidden_size,batch_size,fc_hidden_size,output_size,n_layer)
net.to(device)

criterion=nn.CrossEntropyLoss() #交叉熵损失函数
optimizer=optim.Adam(net.parameters(),lr=0.01)

train_loss_list=[]
train_accu_list=[]
l2_lambda = 0.0002
epoch_num=50
valid_accu_list=[]

for epoch in range(epoch_num):
    net.train().to(device)
    # if(epoch>=50 and epoch%10==9):
    #     optimizer.param_groups[0]['lr']*=0.5
    running_loss=0.0
    correct=0.0
    num=0.0
    curr_epoch=0    
    for input,label in train_loader:
        #文本转为token 
        intoken=[]
        for bat in input:
            token_bat=[]
            for word in bat:
                token_bat.append(iztov[sztoi[word]].tolist())
            intoken.append(token_bat)

        intoken=torch.tensor(intoken)
        intoken,label=intoken.to(device),label.to(device)   #自动写成元组

        #隐藏层还原
        optimizer.zero_grad()
        net.hidd=net.init_hidd()
        net.cell=net.init_cell()
        l2_regularization = sum(torch.sum(torch.pow(param, 2)) for param in net.parameters()) #这个也是torch!
        
        out=net(intoken)   #默认对各批的通道向量作处理

        running_loss+=(criterion(out,label)+l2_lambda*l2_regularization).item()
        loss=criterion(out,label)+l2_lambda*l2_regularization
        loss.backward()
        optimizer.step()

        _,pred_index=torch.max(out,dim=1)        
        _,real_index=torch.max(label,dim=1)
        # print(f'{pred_index<5},{real_index<5}')
        correct+=(pred_index==real_index).sum().item()
        # print(correct)
        num+=label.size(0)
        # print('finish a batch')

    print('epoch:%d;    loss:%.6f;  accu:%.6f'%(epoch+1,running_loss/num,correct/num))
    train_loss_list.append(running_loss/num)
    train_accu_list.append(correct/num)

    if(epoch%5==4):
        net.eval().to(device)
        correct=0.0
        total=0.0    
        with torch.no_grad():   #切断上下文求导计算
            for input,label in test_loader:
                intoken=[]
                for bat in input:
                    token_bat=[]
                    for word in bat:
                        token_bat.append(iztov[sztoi[word]].tolist())
                    intoken.append(token_bat)

                intoken=torch.tensor(intoken)
                intoken,label=intoken.to(device),label.to(device)   #自动写成元组
                # print(f'device={intoken.device,label.device}')
                
                net.hidd=net.init_hidd().to(device)
                net.cell=net.init_cell().to(device)

                out=net(intoken)
                # print(out,label)
                # print(f'{out.shape,label.shape}')
                _,pred_index=torch.max(out,dim=1)        
                _,real_index=torch.max(label,dim=1)
                correct+=(pred_index==real_index).sum().item()
                total+=label.size(0)
            print(f'total={total}')
            print('Accuracy of the network on the test dataset: %d %%'%(100*correct/total))
            valid_accu_list.append(correct/total)

plt.suptitle("lstm training loss and result_0.0")
ax1=plt.subplot(121)
ax1.plot(train_loss_list,label='loss')
ax1.set_xlabel('epoch_num')
ax1.set_ylabel('loss')
ax1.legend()

ax2=plt.subplot(122)
ax2.plot(train_accu_list,label='accu')
ax2.set_xlabel('epoch_num')
ax2.set_ylabel('accu')
ax2.legend()

plt.savefig('lstm.png')

torch.save(net.state_dict(),'lr.params')