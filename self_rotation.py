import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import prepare_for_cifar10 as pf
import resnet
import argparse
import os
import time
import rotation_loss
from pdb import set_trace as bp

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")                  #소문자,resnet으로 시작 하는것으로 model_names에 넣음
                     and callable(resnet.__dict__[name]))
  #CUDA_VISIBLE_DEVICES=0,1,2,3 python self_rotation.py --arch resnetself --save-dir ./save_dir/
parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',           #metavar : description
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')           
parser.add_argument('--save-dir', dest='save_dir',                                       #dest : argument가 저장되는 곳을 지정 args.save_dir에 a가 저장됨 (--save -dir a라고 썻을때)
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--range-of-lr', nargs='+', type=int,help='1234 1234 1234',default=[100,150])       #--range-of-lr 1234 1234 1234
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')        #체크포인트에서 load할때 필요                    
parser.add_argument('--rotation-var',default='rotation_2')

best_prec1 = 0
def main():
    global args,best_prec1
    args = parser.parse_args()
    if args.arch == 'resnetself':
        rotation = True
    else:
        rotation = False
    print('model:',args.arch)
    print('rotation:',rotation)
    
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):     #./save_dir/
        os.makedirs(args.save_dir)
    start_time = time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(time.time()+32400))
    if not os.path.exists('./log_file'):
        os.makedirs('./log_file')
    file = open('./log_file/'+start_time+'.txt','w')
    file.write('architecture: {0}\n'
                'total epochs: {1}\n'
                'batch size: {2}\n'
                'start learning rate: {3}\n'
                'range of learning rate: {4}\n'
                'rotation: {5}\n'.format(
                    args.arch,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    args.range_of_lr,
                    args.rotation_var
                ))
    file.close()


    
    train_loader,val_loader = pf.prepare_dataset(batch=args.batch_size)

    
    model = nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()
    
    
    if args.resume:
        if os.path.isfile(args.resume):          #args.resume: 체크포인트 path
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.range_of_lr, gamma=0.1, last_epoch=-1)
    
    for epoch in range(args.start_epoch, args.epochs):

        
        
        
        train(train_loader, model, criterion, optimizer, epoch, rotation,rotation_var=args.rotation_var,start_time=start_time)
        scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, optimizer,criterion,rotation,start_time=start_time)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.pth'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'model.pth'))


    
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

def train(train_loader, model, criterion, optimizer, epoch,rotation,rotation_var,start_time):
    """
        Run one train epoch
    """ 
    losses = AverageMeter()
    acc = AverageMeter()
    regression_loss = nn.MSELoss().cuda()
    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        if rotation==True: 
            input_var,target_rot,target_var = rotation_loss.__dict__[args.rotation_var](input_var,target_var)  
            optimizer.zero_grad()
            output, output_rot = model(input_var)
            loss = criterion(output,target_var) + torch.sqrt(regression_loss(output_rot,target_rot))
            output, output_rot = output.float(), output_rot.float()
            prec1 = accuracy(output.data, target_var)[0]
            loss.backward()
            optimizer.step()
            loss = loss.float()
            acc.update(prec1.item(), input_var.size(0))    
            losses.update(loss.item(), input_var.size(0))
        else: 

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
        file = open('./log_file/'+start_time+'.txt','a')
        if i % 50 == 0:
            print('{0} -'
                  ' Epoch: [{1}][{2}/{3}] -'
                  ' learning rate: {4:0.5e} -'
                  ' Loss: {5:0.4f} -'
                  ' acc: {6:0.2f} %'.format(string,
                      epoch, i, len(train_loader),optimizer.param_groups[0]['lr'],
                       losses.val,acc.val))
            
            file.write('{0} -'
                  ' Epoch: [{1}][{2}/{3}] -'
                  ' learning rate: {4:0.5e} -'
                  ' Loss: {5:0.4f} -'
                  ' acc: {6:0.2f} %\n'.format(string,
                      epoch, i, len(train_loader),optimizer.param_groups[0]['lr'],
                       losses.val,acc.val))           
    print('average training accuracy: {acc.avg:.3f}'
          .format(acc=acc))                   
    file.write('average training accuracy: {acc.avg:.3f}\n'
          .format(acc=acc))
    file.close()

def validate(val_loader, model,optimizer,criterion,rotation,start_time):
    """
    Run evaluation
    """
    losses = AverageMeter()
    acc = AverageMeter()
    regression_loss = nn.MSELoss().cuda()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            
            if rotation == True:
                output, output_rot = model(input_var)
                
            else:    
                # compute output
                output = model(input_var)
                
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
            file = open('./log_file/'+start_time+'.txt','a')
            if i % 50 == 0:
                print('{0} -'
                    ' Epoch: [{1}/{2}] -'
                    ' learning rate: {3:0.5e} -'
                    ' Loss: {4:0.4f} -'
                    ' acc: {5:0.2f} %'.format(string,
                        i, len(val_loader),optimizer.param_groups[0]['lr'],
                        losses.val,acc.val))
                file.write('{0} -'
                    ' Epoch: [{1}/{2}] -'
                    ' learning rate: {3:0.5e} -'
                    ' Loss: {4:0.4f} -'
                    ' acc: {5:0.2f} %\n'.format(string,
                        i, len(val_loader),optimizer.param_groups[0]['lr'],
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
class AverageMeter(object):   #average sum 등등 만들어주는 클래스
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)  # 1
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res







if __name__ == '__main__':              #이 파일을 직접실행했을때만 main() 함수를 실행시켜라
    main()                          
