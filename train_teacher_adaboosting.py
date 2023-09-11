import argparse
import math
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import time
import copy
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import models
from torch.backends import cudnn

parser = argparse.ArgumentParser(description='PyTorch Dermnet Training')
parser.add_argument('--data', default='/home/s316/workspace/xionggl/data/Dermnet23/', type=str, help='Dataset directory')
parser.add_argument('--arch', default='resnet34_dermnet_aux', type=str, help='network architecture')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
# parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=4, help='number workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=6)
parser.add_argument('--freezed', action='store_true', help='freezing backbone')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoints', type=str, help='checkpoint dir')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = args.data
checkpoint_dir = args.checkpoint_dir

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
cudnn.deterministic = True

log_dir = str(os.path.basename(__file__).split('.')[0]) + '_aux_b64' + '_seed' + str(args.manual_seed)
checkpoint_dir = os.path.join(checkpoint_dir, log_dir)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
class_nums = len(class_names)

net = getattr(models, args.arch)
model = net(num_classes=class_nums).to(device)

if args.freezed:
    # pretrained weights of train_baseline.py
    pretrained_dict = torch.load("/home/s316/workspace/xionggl/project/baseline/checkpoints/xxx.pth.tar")
    model.load_state_dict(pretrained_dict['net'])
    for n,p in model.backbone.named_parameters():
        p.requires_grad = False
    model.backbone.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 70)

log_txt = 'result/' + str(os.path.basename(__file__).split('.')[0]) + '_aux_b64' + '_seed' + str(args.manual_seed) + '.txt'
log_dir = str(os.path.basename(__file__).split('.')[0])

with open(log_txt, 'a+') as f:
    f.write("==========\nArgs:{}\n==========".format(args) + '\n')


def train_model(model, criterion, optimizer, exp_lr_scheduler):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 20)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_loss_cls = 0.0
            running_loss_ada = 0.0
            running_corrects = 0

            list1 = []
            list2 = []
            list3 = []
            list4 = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    logits, ss_logits, feats, w1 = model(inputs, grad=True)
                    loss_cls = criterion(logits[4], labels)

                    _, cls_pred_1 = torch.max(ss_logits[0].data, 1)
                    correct_tensor_1 = cls_pred_1.eq(labels.data.view_as(cls_pred_1))
                    err_1 = (1 - correct_tensor_1.bool().int().float()).dot(torch.squeeze(w1)) / (torch.sum(w1))
                    if err_1 < (1 - (1 / class_nums)):
                        if err_1 == 0:
                            Alpha1 = 6
                            w2 = w1.mul(torch.exp((1 - correct_tensor_1.bool().int().float()) * Alpha1).unsqueeze(1))
                        else:
                            Alpha1 = torch.log((1 - err_1) / err_1) + math.log(class_nums - 1)
                            w2 = w1.mul(torch.exp((1 - correct_tensor_1.bool().int().float()) * Alpha1).unsqueeze(1))
                    else:
                        Alpha1 = 0
                        w2 = w1
                    if Alpha1 == 0 or Alpha1 == 6:
                        list1.append(Alpha1)
                    else:
                        list1.append(Alpha1.item())

                    _, cls_pred_2 = torch.max(ss_logits[1].data, dim=1)
                    correct_tensor_2 = cls_pred_2.eq(labels.data.view_as(cls_pred_2))
                    err_2 = (1 - correct_tensor_2.bool().int().float()).dot(torch.squeeze(w2)) / (torch.sum(w2))
                    if err_2 < (1 - (1 / class_nums)):
                        if err_2 == 0:
                            Alpha2 = 6
                            w3 = w2.mul(torch.exp((1 - correct_tensor_2.bool().int().float()) * Alpha2).unsqueeze(1))
                        else:
                            Alpha2 = torch.log((1 - err_2) / err_2) + math.log(class_nums - 1)
                            w3 = w2.mul(torch.exp((1 - correct_tensor_2.bool().int().float()) * Alpha2).unsqueeze(1))
                    else:
                        Alpha2 = 0
                        w3 = w2
                    if Alpha2 == 0 or Alpha2 == 6:
                        list2.append(Alpha2)
                    else:
                        list2.append(Alpha2.item())

                    _, cls_pred_3 = torch.max(ss_logits[2].data, dim=1)
                    correct_tensor_3 = cls_pred_3.eq(labels.data.view_as(cls_pred_3))
                    err_3 = (1 - correct_tensor_3.bool().int().float()).dot(torch.squeeze(w3)) / (torch.sum(w3))
                    if err_3 < (1 - (1 / class_nums)):
                        if err_3 == 0:
                            Alpha3 = 6
                            w4 = w3.mul(torch.exp((1 - correct_tensor_3.bool().int().float()) * Alpha3).unsqueeze(1))
                        else:
                            Alpha3 = torch.log((1 - err_3) / err_3) + math.log(class_nums - 1)
                            w4 = w3.mul(torch.exp((1 - correct_tensor_3.bool().int().float()) * Alpha3).unsqueeze(1))
                    else:
                        Alpha3 = 0
                        w4 = w3
                    if Alpha3 == 0 or Alpha3 == 6:
                        list3.append(Alpha3)
                    else:
                        list3.append(Alpha3.item())

                    _, cls_pred_4 = torch.max(ss_logits[3].data, dim=1)
                    correct_tensor_4 = cls_pred_4.eq(labels.data.view_as(cls_pred_4))
                    err_4 = (1 - correct_tensor_4.bool().int().float()).dot(torch.squeeze(w4)) / (torch.sum(w4))
                    if err_4 < (1 - (1 / class_nums)):
                        if err_4 == 0:
                            Alpha4 = 6
                        else:
                            Alpha4 = torch.log((1 - err_4) / err_4) + math.log(class_nums - 1)
                            w5 = w4.mul(torch.exp((1 - correct_tensor_4.bool().int().float()) * Alpha4).unsqueeze(1))
                    else:
                        Alpha4 = 0
                        w5 = w4
                    if Alpha4 == 0 or Alpha4 == 6:
                        list4.append(Alpha4)
                    else:
                        list4.append(Alpha4.item())

                    print(len(list1), len(list2), len(list3), len(list4))
                    beta1 = sum(list1) / len(list1)
                    beta2 = sum(list2) / len(list2)
                    beta3 = sum(list3) / len(list3)
                    beta4 = sum(list4) / len(list4)
                    print(beta1, beta2, beta3, beta4)

                    pre_14 = beta1 * ss_logits[0] + beta2 * ss_logits[1] + beta3 * ss_logits[2] + beta4 * ss_logits[3]
                    loss_ada = criterion(pre_14, labels)
                    loss = loss_cls + 0.6 * loss_ada

                    _, preds = torch.max(pre_14, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_loss_cls += loss_cls.item() * inputs.size(0)
                running_loss_ada += loss_ada.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()
                epoch_loss_ada = running_loss_ada / dataset_sizes[phase]

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss_cls = running_loss_cls / dataset_sizes[phase]

            if phase == 'train':
                print('Train Loss: {:.4f} Loss_cls: {:.4f} Loss_ada: {:.4f} Acc: {:.4f}'
                      .format(epoch_loss, epoch_loss_cls, epoch_loss_ada, epoch_acc))
                with open(log_txt, 'a+') as f:
                    f.write('Epoch: {}\t Train Loss: {:.4f}\t Loss_cls: {:.4f}\t Loss_ada: {:.4f}\t Acc: {:.4f}\t'
                            .format(epoch, epoch_loss, epoch_loss_cls, epoch_loss_ada, epoch_acc))
            else:
                print('Test Loss: {:.4f} Loss_cls: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_loss_cls, epoch_acc))
                with open(log_txt, 'a+') as f:
                    f.write('Test Loss: {:.4f}\t Loss_cls: {:.4f}\t Acc: {:.4f}\n'.format(epoch_loss, epoch_loss_cls,
                                                                                          epoch_acc))

            state = {
                'net': model.state_dict(),
                'acc': epoch_acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(checkpoint_dir, 'resnet34_aux' + '.pth.tar'))

            is_best = False
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(checkpoint_dir, 'resnet34_aux' + '.pth.tar'),
                                os.path.join(checkpoint_dir, 'resnet34_aux' + '_best.pth.tar'))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    with open(log_txt, 'a+') as f:
        f.write('Training complete in {:.0f}m\t {:.0f}s\n Best test Acc: {:4f}\n'.format(time_elapsed // 60,
                                                                                         time_elapsed % 60, best_acc))

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    train_model(model, criterion, optimizer, exp_lr_scheduler)
