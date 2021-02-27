import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.nn.init as init

def prepare_dataset(batch):
        
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    transform_train = transforms.Compose([
            
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) #mean,std
            ])
    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=True,download=True,transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch,shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False)
    return trainloader,testloader
