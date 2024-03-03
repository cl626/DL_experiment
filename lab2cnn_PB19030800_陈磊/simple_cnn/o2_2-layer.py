#opt_relu_drop_bn_extc
import torch 
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils

transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]  #mean,std
)

dataset1=torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)

train_size=int(0.8*(len(dataset1)))
valid_size=len(dataset1)-train_size

trainset,validset=torch.utils.data.random_split(dataset1,[train_size,valid_size])

trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)
validloader=torch.utils.data.DataLoader(validset,batch_size=4,shuffle=True,num_workers=0)

testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform)

testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=0)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

import torch.nn as nn
import torch.nn.functional as F
device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

#LeNet
class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        #nn.Sigmoid是一个类，先要赋值为对象
        self.acti=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)  #in_channels为输入的通道
        #32-4=28
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        #28/2=14
        self.conv2=nn.Conv2d(in_channels=16,out_channels=36,kernel_size=3,stride=1)
        #14-2=12
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        #12/2=6
        self.fc1=nn.Linear(1296,128)
        self.fc2=nn.Linear(128,96)
        self.fc3=nn.Linear(96,10)
        self.fc4=nn.Linear(128,10)
        self.bn1=nn.BatchNorm2d(num_features=16)
        self.bn2=nn.BatchNorm2d(num_features=36)
        self.bn3=nn.BatchNorm1d(num_features=128)
        self.bn4=nn.BatchNorm1d(num_features=96)

    def forward(self,x,p):
        #softmax用于多分类
        dropout=nn.Dropout(p)
        x=dropout(self.bn1(self.conv1(x)))    #激活前批正则化
        x=self.pool1(self.acti(x))  #4*6*28*28
        x=dropout(self.bn2(self.conv2(x)))
        x=self.pool2(self.acti(x))
        x=x.view(-1,1296)
        x=self.fc4(self.acti(self.fc1(x)))
        return x

    def dropforward(self,x,p):
        x=self.bn1(self.conv1(x))*(1-p)    #激活前批正则化
        x=self.pool1(self.acti(x))  #4*6*28*28
        x=self.bn2(self.conv2(x))*(1-p)
        x=self.pool2(self.acti(x))
        x=x.view(-1,1296)
        x=self.fc4((self.acti(self.fc1(x))))
        return x
    

net=Lenet5()
net=net.to(device)
print(net)
import torch.optim as optim

criterion=nn.CrossEntropyLoss() #交叉熵损失函数
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

train_loss_list=[]
train_accu_list=[]
l2_lambda = 0.01
epoch_num=50
p=0.5

for epoch in range(epoch_num):
    if(epoch%5==4):
        optimizer.param_groups[0]['lr']*=0.5
    running_loss=0.0
    correct=0.0
    num=0.0
    for i,data in enumerate(trainloader):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)   #自动写成元组
        l2_regularization = sum(torch.sum(torch.pow(param, 2)) for param in net.parameters()) 
        optimizer.zero_grad()

        outputs=net(inputs,p)    #默认对各批的通道向量作处理
        running_loss+=criterion(outputs,labels).item()
        loss=criterion(outputs,labels)+l2_lambda*l2_regularization   
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1)   #gradient clip
        optimizer.step()
        _,predicted =torch.max(outputs.data,1)
        
        correct+=(predicted==labels).sum().item()    
        num+=labels.size(0)

        #if i%2000==1999:
        #    print('[%d,%5d]loss:%.3f'%(epoch+1,i+1,running_loss/2000))
        #    running_loss=0.0
#    print('Finished Training')
    print('epoch:%d;    loss:%.3f'%(epoch+1,running_loss/num))
    train_loss_list.append(running_loss/num)
    train_accu_list.append(correct/num)

#print(len(train_loss_list))
plt.plot(np.arange(1,epoch_num+1,1),train_loss_list,'g',label='train_loss')
plt.plot(np.arange(1,epoch_num+1,1),train_accu_list,'r',label='train_acc')
plt.xlabel('epoch')
plt.title('average loss,acc-epoch')
plt.savefig('Lenet-5.png')

correct=0
total=0

with torch.no_grad():   #切断上下文求导计算
    for data in validloader:
        images,labels=data
        images,labels=images.to(device),labels.to(device)
        outputs=net.dropforward(images,p)
        _,predicted =torch.max(outputs.data,1)  #return 维度1上的最大值及其索引
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    
print('Accuracy of the network on the 10000 valid images: %d %%'%(100*correct/total))