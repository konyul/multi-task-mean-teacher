#!/bin/sh


python3 student.py --dataset cifar10 --arch resnetself --auxiliary rotation --augmentation 4
python3 student.py --dataset cifar100 --arch resnet32
python3 student.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation 4
python3 teacher.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation 4 --teacher
python3 teacher.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation 2 --teacher



python3 alarm.py
