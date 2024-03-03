import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np

device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
dp=0.5
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), #长宽=32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), #32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),    #32/2=16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   #16/2=8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   #8/2=4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #4/2=2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #2/2=1
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dp),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dp),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def valid_forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier[1](self.classifier[0](x))*(1-dp)
        x = self.classifier[4](self.classifier[3](x))*(1-dp)
        x = self.classifier[6](x)
        return x
        
batch_size = 128
learning_rate = 0.001
num_epochs = 50
# 数据增强
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
dataset1 = datasets.CIFAR10(root='./data', train=True, download=False,transform=train_transforms)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False,transform=test_transforms)

train_size=int(0.8*len(dataset1))
valid_size=len(dataset1)-train_size

# 加载数据集到生成器中
trainset,validset=torch.utils.data.random_split(dataset1,[train_size,valid_size])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
valid_loader =torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net=VGG()
net=net.to(device)
print(net)
import torch.optim as optim

criterion=nn.CrossEntropyLoss() #交叉熵损失函数
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

train_loss_list=[]
train_accu_list=[]
l2_lambda = 0.01
epoch_num=30
valid_accu_list=[]
for epoch in range(epoch_num):
    if(epoch%5==4):
        optimizer.param_groups[0]['lr']*=0.5
    running_loss=0.0
    correct=0.0
    num=0.0
    for i,data in enumerate(train_loader):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)   #自动写成元组
        l2_regularization = sum(torch.sum(torch.pow(param, 2)) for param in net.parameters()) 
        optimizer.zero_grad()

        outputs=net(inputs)    #默认对各批的通道向量作处理
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
    if(epoch%10==9):
        correct=0
        total=0    
        with torch.no_grad():   #切断上下文求导计算
            for data in valid_loader:
                images,labels=data
                images,labels=images.to(device),labels.to(device)
                outputs=net.valid_forward(images)
                _,predicted =torch.max(outputs.data,1)  #return 维度1上的最大值及其索引
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()        
        print('Accuracy of the network on the 10000 valid images: %d %%'%(100*correct/total))
        valid_accu_list.append(correct/total)
    
#print(len(train_loss_list))
plt.plot(np.arange(1,epoch_num+1,1),train_loss_list,'g',label='train_loss')
plt.plot(np.arange(1,epoch_num+1,1),train_accu_list,'b',label='train_acc')
plt.plot(np.arange(10,epoch_num+1,10),valid_accu_list,'r',label='valid_acc')
plt.xlabel('epoch')
plt.title('dp5 average loss,acc-epoch')
plt.legend()
plt.savefig('VGGdp5.png')