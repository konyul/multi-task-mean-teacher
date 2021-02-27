import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.nn.init as init
__all__ = ['rotation_2', 'rotation_4']
def rotation_2(batch_data,batch_target):
    n = batch_data.shape[0]
    rotated_images=[]
    # targets_r = torch.randint(0,4,(n,))
    # targets_r_zero = torch.zeros(n,)                                        #original
    # targets_a = torch.stack([targets_r_zero,targets_r],dim=1).view(-1)

    #targets_r = torch.rand((n,4))       #uniform distribution
    
    targets_r_sub = torch.exp(torch.randn((n,4)))
    targets_r = targets_r_sub/(1+targets_r_sub)                      #normal distribution

    targets_r_zero = torch.zeros((n,4))
    argmx_r = torch.argmax(targets_r,dim=1)                                   #random distribution
    targets_a = torch.stack([targets_r_zero,targets_r],dim=1).view(-1,4)  
    argmx = torch.argmax(targets_a,dim=1)
    targets_a_sub = targets_a.scatter(1,argmx.view(-1,1).long(),1)

    for i in range(n):
        #inputs_rot = torch.rot90(batch_data[i],targets_r[i],[1,2]).reshape(1,3,32,32)
        inputs_rot = torch.rot90(batch_data[i],argmx_r[i],[1,2]).reshape(1,3,32,32)
        rotated_images.append(inputs_rot)
    inputs_r = torch.cat(rotated_images,0)
    size = batch_data.shape[1:]
    result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
    target_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
    target_cls = target_cls.cuda()
    #target_rot = torch.zeros(2*n,4).scatter(1,targets_a.view(-1,1).long(),0.6)+0.1  # 0.7 0.1 0.1 0.1
    #target_rot = torch.zeros(2*n,4).scatter(1,targets_a.view(-1,1).long(),1)   #1 0 0 0
    targets_rot = targets_a_sub * targets_a        #random but ^2 except for max
    targets_rot = targets_rot.cuda()
    return result_input,targets_rot,target_cls



def rotation_4(batch_data,batch_target):
    n = batch_data.shape[0]
    size = batch_data.shape[1:]
    result_input = torch.stack([torch.rot90(batch_data, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
    target_cls = torch.stack([batch_target for i in range(4)], 1).view(-1)
    target_rot = torch.stack([torch.tensor([0,1,2,3]) for i in range(n)], 0).view(-1)
    target_rot = target_rot.cuda()

    return result_input,target_rot,target_cls 