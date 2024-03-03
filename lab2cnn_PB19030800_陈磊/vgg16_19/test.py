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
dp=0.0
cfg={
    'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'VGG19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
}
class VGG(nn.Module):
    def __init__(self,vgg_name):
        super(VGG, self).__init__()
        self.features=self._make_layers(cfg[vgg_name])
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

    def _make_layers(self,cfg):
        layers=[]
        in_channels=3
        for x in cfg:
            if x=='M':
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers+=[nn.Conv2d(in_channels,x,kernel_size=3,padding=1),
                         nn.BatchNorm2d(x),nn.ReLU(inplace=True)]
                in_channels=x
        layers+=[nn.AvgPool2d(kernel_size=1,stride=1)]
        return nn.Sequential(*layers)
        
batch_size = 128
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

net=VGG('VGG19')
net=net.to(device)
#print(net)
import torch.optim as optim

criterion=nn.CrossEntropyLoss() #交叉熵损失函数
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

train_loss_list=[]
train_accu_list=[]
l2_lambda = 0.01
epoch_num=30
valid_accu_list=[]

net.load_state_dict(torch.load('vgg19pic.params'))
correct=0
total=0    
with torch.no_grad():   #切断上下文求导计算
    for data in test_loader:
        images,labels=data
        images,labels=images.to(device),labels.to(device)
        outputs=net.valid_forward(images)
        _,predicted =torch.max(outputs.data,1)  #return 维度1上的最大值及其索引
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()        
print('Accuracy of the network on the 10000 valid images: %d %%'%(100*correct/total))
