import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.nn.init as init
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10,CIFAR100
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler as BaseSampler
import os, csv, random, torch, numpy as np


__all__ = ['cifar10','cifar100']


def cifar10(batch):
        
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5),
                                                                   (0.5, 0.5, 0.5))])
    transform_test  = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5),
                                                                   (0.5, 0.5, 0.5))])

    train_indices = [i for i in range(50000) if i not in np.load('splits/{}_val_idx.npy'.format('cifar10'))]
    val_indices = np.load('splits/{}_val_idx.npy'.format('cifar10'))
    trainset = Subset(CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform_train), train_indices)
    #trainloader = check_dataloader(trainset,args.val_freq,args.batch_size)

    valset = Subset(CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform_test), val_indices)
    valloader = torch.utils.data.DataLoader(valset,batch_size=batch,shuffle=False)
    testset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False)
    return trainset,testloader , valloader

def cifar100(batch):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5),
                                                                   (0.5, 0.5, 0.5))])
    transform_test  = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5),
                                                                   (0.5, 0.5, 0.5))])
    train_indices = [i for i in range(50000) if i not in np.load('splits/{}_val_idx.npy'.format('cifar100'))]
    val_indices = np.load('splits/{}_val_idx.npy'.format('cifar100'))
    trainset = Subset(CIFAR100(root='./CIFAR100', train=True, download=True, transform=transform_train), train_indices)
    #trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch,shuffle=True)

    valset = Subset(CIFAR100(root='./CIFAR100', train=True, download=True, transform=transform_test), val_indices)
    valloader = torch.utils.data.DataLoader(valset,batch_size=batch,shuffle=False)
    testset = torchvision.datasets.CIFAR100(root='./CIFAR100',train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False)
    return trainset,testloader ,valloader



class BatchSampler(BaseSampler):
    def __init__(self, dataset, num_iterations, batch_size):

        self.dataset = dataset
        self.num_iterations = num_iterations
        self.batch_size = batch_size

        self.sampler = None

    def __iter__(self):
        indices = []
        for _ in range(self.num_iterations):
            indices = random.sample(range(len(self.dataset)),
                                    self.batch_size)
            yield indices

    def __len__(self):
        return self.num_iterations



def check_dataloader(dataset, num_iterations, *args):
    sampler = BatchSampler(dataset, num_iterations, *args)
    return DataLoader(dataset, num_workers=12, batch_sampler=sampler)