import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.nn.init as init
from random import *
__all__ = ['color','rotation','rotation_4','color_3','color_6','joint_24','joint']

def color(batch_data,batch_target):
    n = batch_data.shape[0]
    colored_images=[]
    size = batch_data.shape[1:]
    targets_r = torch.randint(0,6,(n,))
    targets_r_zero = torch.zeros(n,)
    targets_a = torch.stack([targets_r_zero,targets_r],dim=1).view(-1)
    for i in range(n):
        sample_input = torch.stack([batch_data[i],
                              torch.stack([batch_data[i, 0, :, :], batch_data[i, 2, :, :], batch_data[i, 1, :, :]], 0),
                              torch.stack([batch_data[i, 1, :, :], batch_data[i, 0, :, :], batch_data[i, 2, :, :]], 0),
                              torch.stack([batch_data[i, 1, :, :], batch_data[i, 2, :, :], batch_data[i, 0, :, :]], 0),
                              torch.stack([batch_data[i, 2, :, :], batch_data[i, 0, :, :], batch_data[i, 1, :, :]], 0),
                              torch.stack([batch_data[i, 2, :, :], batch_data[i, 1, :, :], batch_data[i, 0, :, :]], 0)], 0).view(-1, *size)
        
        inputs_col = sample_input[targets_r[i]]
        colored_images.append(inputs_col)
    inputs_r = torch.stack(colored_images,0)
    
    result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
    targets_col = torch.zeros(2*n,6).scatter(1,targets_a.view(-1,1).long(),1)   #1 0 0 0
    targets_col = targets_col.cuda()
    return result_input,targets_col

def rotation(batch_data,batch_target):
    n = batch_data.shape[0]
    rotated_images=[]
    targets_r = torch.randint(0,4,(n,))
    targets_r_zero = torch.zeros(n,)
    targets_a = torch.stack([targets_r_zero,targets_r],dim=1).view(-1)
    for i in range(n):
        inputs_rot = torch.rot90(batch_data[i],targets_r[i],[1,2]).reshape(1,3,32,32)
        rotated_images.append(inputs_rot)
    inputs_r = torch.cat(rotated_images,0)
    size = batch_data.shape[1:]
    result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
    targets_rot = torch.zeros(2*n,4).scatter(1,targets_a.view(-1,1).long(),1)   #1 0 0 0
    targets_rot = targets_rot.cuda()
    return result_input,targets_rot

def joint(batch_data,batch_target,auxiliary):
    
    n = batch_data.shape[0]
    auxiliary_images = []
    targets_r = torch.randint(0,24,(n,))
    targets_r_zero = torch.zeros(n,)
    size = batch_data.shape[1:]
    targets_a = torch.stack([targets_r_zero,targets_r],dim=1).view(-1)
    for i in range(n):
        augmented_images=[]
        for k in range(4):
            x = torch.rot90(batch_data,k,(2,3))
            augmented_images.append(x[i])
            augmented_images.append(torch.stack([x[i, 0, :, :], x[i, 2, :, :], x[i, 1, :, :]], 0))
            augmented_images.append(torch.stack([x[i, 1, :, :], x[i, 0, :, :], x[i, 2, :, :]], 0))
            augmented_images.append(torch.stack([x[i, 1, :, :], x[i, 2, :, :], x[i, 0, :, :]], 0))
            augmented_images.append(torch.stack([x[i, 2, :, :], x[i, 0, :, :], x[i, 1, :, :]], 0))
            augmented_images.append(torch.stack([x[i, 2, :, :], x[i, 1, :, :], x[i, 0, :, :]], 0))
        augmented_images = torch.stack(augmented_images, 0).view(-1, *size).contiguous()
        auxiliary_images.append(augmented_images[targets_r[i]])
    inputs_a = torch.stack(auxiliary_images,0)
    result_input = torch.stack([batch_data,inputs_a],1).view(-1,*size)
    targets_aux = torch.zeros(2*n,24).scatter(1,targets_a.view(-1,1).long(),1)   #1 0 0 0
    targets_aux = targets_aux.cuda()
    return result_input,targets_aux


def rotation_4(batch_data,batch_target):
    n = batch_data.shape[0]
    size = batch_data.shape[1:]
    result_input = torch.stack([torch.rot90(batch_data, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
    target_rot = torch.stack([torch.tensor([0,1,2,3]) for i in range(n)], 0).view(-1)
    target_rot = torch.zeros(4*n,4).scatter(1,target_rot.view(-1,1).long(),1)   #1 0 0 0
    target_rot = target_rot.cuda()
    #만약에 Softmax가 더 잘나오면
    return result_input,target_rot

def color_3(batch_data,batch_target):
    n = batch_data.shape[0]
    size = batch_data.shape[1:]
    result_input = torch.stack([batch_data,
                              torch.stack([batch_data[:, 1, :, :], batch_data[:, 2, :, :], batch_data[:, 0, :, :]], 1),
                              torch.stack([batch_data[:, 2, :, :], batch_data[:, 0, :, :], batch_data[:, 1, :, :]], 1)], 1).view(-1, *size)
    target_col = torch.stack([torch.tensor([0,1,2]) for i in range(n)], 0).view(-1)
    target_col = torch.zeros(3*n,4).scatter(1,target_col.view(-1,1).long(),1)   #1 0 0 0
    target_col = target_col.cuda()

    return result_input,target_col

def color_6(batch_data,batch_target):
    n = batch_data.shape[0]
    size = batch_data.shape[1:]
    result_input = torch.stack([batch_data,
                              torch.stack([batch_data[:, 0, :, :], batch_data[:, 2, :, :], batch_data[:, 1, :, :]], 1),
                              torch.stack([batch_data[:, 1, :, :], batch_data[:, 0, :, :], batch_data[:, 2, :, :]], 1),
                              torch.stack([batch_data[:, 1, :, :], batch_data[:, 2, :, :], batch_data[:, 0, :, :]], 1),
                              torch.stack([batch_data[:, 2, :, :], batch_data[:, 0, :, :], batch_data[:, 1, :, :]], 1),
                              torch.stack([batch_data[:, 2, :, :], batch_data[:, 1, :, :], batch_data[:, 0, :, :]], 1)], 1).view(-1, *size)
    target_col = torch.stack([torch.tensor([0,1,2,3,4,5]) for i in range(n)], 0).view(-1)
    target_col = torch.zeros(6*n,4).scatter(1,target_col.view(-1,1).long(),1)   #1 0 0 0
    target_col = target_col.cuda()

    return result_input,target_col

def joint_24(batch_data,batch_target,auxiliary):
    n = batch_data.shape[0]
    size = batch_data.shape[1:]
    augmented_images = []
    auxiliary_list = auxiliary.split('_')
    if 'rotation' in auxiliary_list and 'color' in auxiliary_list:
        for k in range(4):
            x = torch.rot90(batch_data,k,(2,3))
            augmented_images.append(x)
            augmented_images.append(torch.stack([x[:, 0, :, :], x[:, 2, :, :], x[:, 1, :, :]], 1))
            augmented_images.append(torch.stack([x[:, 1, :, :], x[:, 0, :, :], x[:, 2, :, :]], 1))
            augmented_images.append(torch.stack([x[:, 1, :, :], x[:, 2, :, :], x[:, 0, :, :]], 1))
            augmented_images.append(torch.stack([x[:, 2, :, :], x[:, 0, :, :], x[:, 1, :, :]], 1))
            augmented_images.append(torch.stack([x[:, 2, :, :], x[:, 1, :, :], x[:, 0, :, :]], 1))
        result_input = torch.stack(augmented_images, 1).view(-1, *size).contiguous()
        target_auxiliary = torch.stack([torch.tensor(range(24)) for i in range(n)], 0).view(-1)
        target_auxiliary = torch.zeros(3*n,4).scatter(1,target_auxiliary.view(-1,1).long(),1)   #1 0 0 0
        target_auxiliary = target_auxiliary.cuda()
    return result_input,target_auxiliary

