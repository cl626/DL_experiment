#可以加载到GPU
import torch
import torch.utils.data as data
import numpy as np
import torch.nn.functional as F
from torch import optim,nn

import pdb

#print(torch.cuda.is_available())
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

class MyDataset(data.Dataset):
    def __init__(self):
        self.x=(torch.tensor(np.random.uniform(1,16,size=5000)).float()).to(device)
        self.y=(torch.log(self.x)/torch.log(torch.tensor(2))+torch.cos(torch.pi*0.5*self.x)).to(device)
#        print(self.x.device)
#        print(self.y.device)

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)

dataset=MyDataset()
train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size
train_dataset,test_dataset=data.random_split(dataset,[train_size,test_size])
train_dataloader=data.DataLoader(train_dataset, batch_size=1)   #不用显示批数
#print(len(train_dataset))
test_dataloader=data.DataLoader(test_dataset, batch_size=1)

class Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,n_hidden_3,out_dim):
        super(Net,self).__init__()
        self.layer1=nn.Linear(in_dim,n_hidden_1)
        self.layer2=nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3=nn.Linear(n_hidden_2,n_hidden_3)
        self.layer4=nn.Linear(n_hidden_3,out_dim)


    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=(x-torch.mean(x))/torch.var(x) #层归一化
        x=F.relu(self.layer2(x))
        x=(x-torch.mean(x))/torch.var(x)
        x=F.relu(self.layer3(x))
        x=(x-torch.mean(x))/torch.var(x)
        x=self.layer4(x)
        return x
#什么时候用softmax,多酚类？

model=Net(1,20,20,30,1)
model.to(device)

criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

losses=[]
acces=[]
eval_losses=[]
eval_acces=[]
num_epochs=150
l2_lambda = 0.01

for epoch in range(num_epochs):
    train_loss=0
    train_acc=0
    if epoch%5==0:
        optimizer.param_groups[0]['lr']*=0.5
    for x,y in train_dataloader:
        out=model(x)
        l2_regularization = sum(torch.sum(torch.pow(param, 2)) for param in model.parameters()) #这个也是torch!
#        print(l2_regularization.size())
        loss=criterion(out,y)+l2_lambda * l2_regularization

        optimizer.zero_grad()
        loss.backward()    #loss梯度下降求导
        nn.utils.clip_grad_norm_(model.parameters(), 1)   #gradient clip
        optimizer.step()    #利用这个导数迭代
        train_loss+=(loss-l2_lambda * l2_regularization).item() #这里要把torch(l2_regulation)转为item()
#        print(loss.item())
        train_acc+=(abs(out*1.0/y)).item()  #转为标量
#        print(type(train_loss))
    losses.append(train_loss/len(train_dataloader))
    acces.append(train_acc/len(train_dataloader))
    
#    print(type(epoch),type(losses[epoch]))
    print('epoch:{},Train Loss:{:4f},Train acc:{:4f}'.format(epoch+1,losses[epoch],acces[epoch]))

eval_loss=0
eval_acc=0

for x,y in test_dataloader:
    out=model(x)
    loss=criterion(out,y)
    eval_loss+=loss.item()
    eval_acc+=(abs(out*1.0/y)).item()
#        print(type(eval_loss))
#eval_losses.append(eval_loss/len(test_dataloader))
#eval_acces.append(eval_acc/len(test_dataloader))
print('Test Loss:{:4f},Test acc:{:4f}'.format(eval_loss/len(test_dataloader),eval_acc/len(test_dataloader)))

import matplotlib.pyplot as plt


plt.plot(list(range(1,num_epochs+1,1)),losses,color='b',label='test_loss')
plt.xlabel('epoch')
plt.ylabel('loss amount')
plt.legend()
plt.title('loss-epoch')
plt.savefig('20_20_30_5000_150.png')
plt.show()