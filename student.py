import torch, torchvision
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torchvision.transforms as transforms
import datasets, cresnet, augmentations
import os, time, argparse, math
import torch.nn.functional as F
from utils import AverageMeter, accuracy, sigmoid_rampup
from torch.autograd import Variable
from datasets import check_dataloader

parser = argparse.ArgumentParser(description='Multi-task learning')

parser.add_argument('--dataset',required=True,default=False)           #cifar10,cifar100
parser.add_argument('--arch', '-a',required=True, metavar='ARCH', default='resnetself')           #metavar : description
parser.add_argument('--auxiliary',default=False) #rotation color exemplar      if joint, rotation_color
parser.add_argument('--augmentation',type=int,default=False) # 4 or 2 in rotation, 3 6 2 in color permutation


parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--range-of-lr', nargs='+', type=int,help='1234 1234 1234',default=[40,60])       #--range-of-lr 1234 1234 1234                    
parser.add_argument('--ema-decay',type=float,default=0.999) 
parser.add_argument('--ema-class-loss',action='store_true') #auxiliary
parser.add_argument('--val-freq', type=int, default=1000)
parser.add_argument('--teacher', action='store_true')

best_prec1 = 0
args = parser.parse_args()
def main():
    
    global args,best_prec1
    
    log_directory = [args.dataset,args.arch,args.auxiliary,args.augmentation]
    while False in log_directory:
        log_directory.remove(False)
    for i in range(len(log_directory)):
        log_directory[i] = str(log_directory[i])
    log_directory = './logs/'+'/'.join(log_directory) + '/'
 
    

    start_time = time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(time.time()))
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    file = open(log_directory+start_time+'.txt','w')

    
    file.write('architecture: {0}\n'
                'total epochs: {1}\n'
                'batch size: {2}\n'
                'start learning rate: {3}\n'
                'range of learning rate: {4}\n'
                'dataset: {5}\n'
                'auxiliary type: {6}\n'
                'number of augmentation: {7}\n'
                'ema-auxiliary-loss: {8}\n'
                'teacher: {9}'
                .format(
                    args.arch,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    args.range_of_lr,
                    args.dataset,
                    args.auxiliary,
                    args.augmentation,
                    args.ema_class_loss,
                    args.teacher
                    
                ))
    file.close()
    
    trainset,testloader, valloader = datasets.__dict__[args.dataset](batch=args.batch_size)

    trainloader = check_dataloader(trainset,args.val_freq,args.batch_size)

    

    
    if args.auxiliary =='rotation':
        auxiliary_classes = 4
    elif args.auxiliary =='color' and args.augmentation == 2:
        auxiliary_classes = 6
    elif args.auxiliary == 'color' and args.augmentation == 3:
        auxiliary_classes = 3
    elif args.auxiliary == 'color' and args.augmentation == 6:
        auxiliary_classes = 6

    



    if args.arch == 'resnetself':
            model = nn.DataParallel(cresnet.__dict__[args.arch](num_classes=int(args.dataset[5:]),num_auxiliary_classes=auxiliary_classes)).cuda()
    else:
        model = nn.DataParallel(cresnet.__dict__[args.arch](num_classes=int(args.dataset[5:]))).cuda()
        
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.range_of_lr, gamma=0.1, last_epoch=-1)
    
    for epoch in range(args.epochs):
        s = time.time()
        train(trainloader, model, criterion, optimizer,epoch,auxiliary=args.auxiliary,augmentation=args.augmentation,start_time=start_time,log_directory=log_directory)
        scheduler.step()
        

                
        # evaluate on validation set
        print("Evaluating the primary model:")
        prec1 = validate(testloader, model, optimizer,criterion,auxiliary=args.auxiliary,start_time=start_time,log_directory=log_directory)
        
        
    
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict':model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(log_directory, 'checkpoint.pth'))

        save_checkpoint({
            'state_dict':model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(log_directory, 'model.pth'))



def train(trainloader, model, criterion, optimizer,epoch,auxiliary,augmentation,start_time,log_directory):
    


    losses = AverageMeter()
    acc = AverageMeter()
    acc_aux = AverageMeter()
    regression_loss = nn.MSELoss().cuda()
    # switch to train mode
    model.train()
   

    for i, (input, target) in enumerate(trainloader):

        # measure data loading time
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        batch_size =input_var.shape[0]
        if auxiliary == 'rotation':
            if augmentation == 2:
                input_var,target_aux = augmentations.__dict__['rotation'](input_var,target_var)
            elif augmentation == 4:
                input_var,target_aux = augmentations.__dict__['rotation_4'](input_var,target_var)
            else:
                assert False, 'choose augmentation'

            num_aug = input_var.shape[0] // batch_size
            optimizer.zero_grad() 
            output, output_aux = model(input_var)
            output = output[::num_aug]
            soft = F.log_softmax(output_aux,dim=1)
            loss = criterion(output,target_var) + torch.mean(-torch.sum(target_aux*soft,dim=1))
            output, output_aux= output.float(), output_aux.float()
            prec1 = accuracy(output.data, target_var)[0]
            prec_aux = accuracy(output_aux.data,torch.argmax(target_aux,dim=1))[0]
            acc_aux.update(prec_aux.item(),input_var.size(0))    
            loss.backward()
            optimizer.step()
            loss = loss.float()
            acc.update(prec1.item(), input_var.size(0))
            losses.update(loss.item(), input_var.size(0))
            


        elif auxiliary == 'color':
            if augmentation == 2:
                input_var,target_aux = augmentations.__dict__['color'](input_var,target_var)
                
            elif augmentation == 6:
                input_var,target_aux = augmentations.__dict__['color_6'](input_var,target_var)  
                
            elif augmentation == 3:
                input_var,target_aux = augmentations.__dict__['color_3'](input_var,target_var)  
            num_aug = input_var.shape[0] // batch_size
            optimizer.zero_grad() 
            output, output_aux = model(input_var)
            output = output[::num_aug]
            soft = F.log_softmax(output_aux,dim=1)
            loss = criterion(output,target_var) + torch.mean(-torch.sum(target_aux*soft,dim=1))
            output, output_aux= output.float(), output_aux.float()
            prec1 = accuracy(output.data, target_var)[0]
            prec_aux = accuracy(output_aux.data,torch.argmax(target_aux,dim=1))[0]
            acc_aux.update(prec_aux.item(),input_var.size(0))    
            loss.backward()
            optimizer.step()
            loss = loss.float()
            acc.update(prec1.item(), input_var.size(0))
            losses.update(loss.item(), input_var.size(0))
        elif auxiliary == False: 

            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            output = output.float()
            prec1 = accuracy(output.data, target)[0]   
            loss.backward()
            optimizer.step()
            loss = loss.float()
            acc.update(prec1.item(), input_var.size(0))    
            losses.update(loss.item(), input_var.size(0))
            
        # measure elapsed time
        tm = time.localtime(time.time())
        string = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
        file = open(log_directory+start_time+'.txt','a')
        if i % 50 == 0:
            print('{0} -'
                  ' Epoch: [{1}][{2}/{3}] -'
                  ' learning rate: {4:0.5e} -'
                  ' Loss: {5:0.4f} -'
                  ' main acc: {6:0.2f} %'
                  ' auxiliary acc: {7:0.2f} %'.format(string,
                      epoch, i, len(trainloader),optimizer.param_groups[0]['lr'],
                       losses.val,acc.val,acc_aux.val))
            
            file.write('{0} -'
                  ' Epoch: [{1}][{2}/{3}] -'
                  ' learning rate: {4:0.5e} -'
                  ' Loss: {5:0.4f} -'
                  ' acc: {6:0.2f} %'
                  ' auxiliary acc: {7:0.2f} %\n'.format(string,
                      epoch, i, len(trainloader),optimizer.param_groups[0]['lr'],
                       losses.val,acc.val,acc_aux.val))           
    print('average training accuracy: {acc.avg:.3f}'
          .format(acc=acc))                   
    file.write('average training accuracy: {acc.avg:.3f}\n'
          .format(acc=acc))
    file.close()

def validate(testloader, model,optimizer,criterion,auxiliary,start_time,log_directory):
    """
    Run evaluation
    """
    losses = AverageMeter()
    acc = AverageMeter()
    acc_aux = AverageMeter()
    regression_loss = nn.MSELoss().cuda()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(testloader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            
            if auxiliary == False:    
                # compute output
                output = model(input_var)
                
            else:
                output, output_aux = model(input_var)
                
            loss = criterion(output,target_var)
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            acc.update(prec1.item(), input.size(0))
                  

            # measure elapsed time
            tm = time.localtime(time.time())
            string = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
            file = open(log_directory+start_time+'.txt','a')
            if i % 50 == 0:
                print('{0} -'
                    ' Epoch: [{1}/{2}] -'
                    ' learning rate: {3:0.5e} -'
                    ' Loss: {4:0.4f} -'
                    ' acc: {5:0.2f} %'.format(string,
                        i, len(testloader),optimizer.param_groups[0]['lr'],
                        losses.val,acc.val))
                file.write('{0} -'
                    ' Epoch: [{1}/{2}] -'
                    ' learning rate: {3:0.5e} -'
                    ' Loss: {4:0.4f} -'
                    ' acc: {5:0.2f} %\n'.format(string,
                        i, len(testloader),optimizer.param_groups[0]['lr'],
                        losses.val,acc.val))        
    print('average validation accuracy: {acc.avg:.3f}'
          .format(acc=acc))
    file.write('---------------------------------------------\n'
               '|      average validation accuracy          |\n'
               '|             {acc.avg:.3f}                        |\n'
               '---------------------------------------------\n'
          .format(acc=acc))     
    file.close()
    return acc.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)





if __name__ == '__main__':              #이 파일을 직접실행했을때만 main() 함수를 실행시켜라
    main()                          
