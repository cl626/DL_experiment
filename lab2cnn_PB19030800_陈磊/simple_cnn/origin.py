import torch 
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils

transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]  #mean,std
)

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)

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
        self.pool1=nn.AvgPool2d(kernel_size=2,stride=2)
        #28/2=14
        self.conv2=nn.Conv2d(in_channels=16,out_channels=36,kernel_size=3,stride=1)
        #14-4=12
        self.pool2=nn.AvgPool2d(kernel_size=2,stride=2)
        #12/2=6
        self.fc1=nn.Linear(1296,128)
        self.fc2=nn.Linear(128,96)
        self.fc3=nn.Linear(96,10)

    def forward(self,x):
        #softmax用于多分类
        x=self.pool1(self.acti(self.conv1(x)))
        x=self.pool2(self.acti(self.conv2(x)))
        x=x.view(-1,1296)
        x=self.fc3(self.acti(self.fc2(self.acti(self.fc1(x)))))
        return x

net=Lenet5()
net=net.to(device)
print(net)
import torch.optim as optim

criterion=nn.CrossEntropyLoss() #交叉熵损失函数
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
train_loss_list=[]
train_accu_list=[]

for epoch in range(10):
    running_loss=0.0
    correct=0.0
    for i,data in enumerate(trainloader):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)   #自动写成元组

        optimizer.zero_grad()

        outputs=net(inputs)    #默认对各批的通道向量作处理
        loss=criterion(outputs,labels)    
        loss.backward()
        optimizer.step()
        _,predicted =torch.max(outputs.data,1)
        
        correct+=(predicted==labels).sum().item()
        running_loss+=loss.item()
        if i%2000==1999:
            print('[%d,%5d]loss:%.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss=0.0
    print('Finished Training')
    train_loss_list.append(running_loss/len(trainloader))
    train_accu_list.append(correct/len(trainloader))

plt.plot(np.arange(1,11,1),train_loss_list,'g',label='train_loss')
plt.plot(np.arange(1,11,1),train_accu_list,'r',label='train_acc')
plt.xlabel('epoch')
plt.title('average loss,acc-epoch')
plt.legend()
plt.savefig('Lenet-5.png')
correct=0
total=0

with torch.no_grad():   #切断上下文求导计算
    for data in testloader:
        images,labels=data
        images,labels=images.to(device),labels.to(device)
        outputs=net(images)
        _,predicted =torch.max(outputs.data,1)  #return 维度1上的最大值及其索引
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    
print('Accuracy of the network on the 10000 test images: %d %%'%(100*correct/total))