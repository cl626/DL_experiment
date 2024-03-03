import torch 
import torchvision
import torchvision.transforms as transforms

transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]  #mean,std
)

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)

testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform)

testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=0)

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
#    plt.savefig('testpic/pic32')

dataiter=iter(testloader)
images,labels=next(dataiter)

print(labels)
print(labels.size(0))
print(images.shape)
image1=images[0]
tf1=transforms.Compose([
    transforms.CenterCrop(10),
    transforms.RandomCrop(10,padding=0)
])
tf2=transforms.Compose([
    transforms.CenterCrop(10),
    transforms.RandomCrop(10,padding=0)
])
tcc=transforms.CenterCrop(10)
trc=transforms.RandomCrop(5)
image2=tf1(image1)
image3=tf2(image1)
image4=tcc(image1)
image5=tcc(image4)
imshow(torchvision.utils.make_grid(image5))