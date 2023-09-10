import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import time
import copy
import models
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchmetrics
import argparse
from torch.backends import cudnn


parser = argparse.ArgumentParser(description='PyTorch Dermnet Training')
parser.add_argument('--data', default='/home/s316/workspace/xionggl/data/23/', type=str,
                    help='Dataset directory')
parser.add_argument('--arch', default='resnet18_dermnet', type=str, help='network architecture')

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
# parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=4, help='number workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=3407)
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str, help='checkpoint dir')


args = parser.parse_args()

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
cudnn.deterministic = True
cudnn.benchmark = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint_dir = args.checkpoint_dir
log_dir = str(os.path.basename(__file__).split('.')[0]) + '_b64' + '_seed' + str(args.manual_seed)
checkpoint_dir = os.path.join(checkpoint_dir, log_dir)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

log_txt = 'result/' + str(os.path.basename(__file__).split('.')[0]) + '_b64' + '_seed' + str(args.manual_seed) + '.txt'
log_dir = str(os.path.basename(__file__).split('.')[0])

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

image_datasets = {x: datasets.ImageFolder(os.path.join(args.data, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in
               ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
class_nums = len(class_names)

model = getattr(models, args.arch)

model = net(num_classes=class_nums).to(device)

model_dict = model.state_dict()
pretrained_dict = torch.load("/home/s316/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth")
pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc.weight' not in k and 'fc.bias' not in k)}
model_dict.update(pretrained_dict1)
model.load_state_dict(model_dict)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 70)



def train_model(model, criterion, optimizer, exp_lr_scheduler):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 20)
        
        recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=class_nums).to(device)
        precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=class_nums).to(device)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    recall(preds, labels)
                    precision(preds, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                exp_lr_scheduler.step()
            
            
            total_recall = recall.compute()
            total_precision = precision.compute()

            recall_value = total_recall.item()
            precision_value = total_precision.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} Recall: {:.4f} precision: {:.4f}'.format(phase, epoch_loss, epoch_acc, recall_value, precision_value))

            if phase == 'train':
                with open(log_txt, 'a+') as f:
                    f.write('Epoch: {}\t {} Loss: {:.4f}\t Acc: {:.4f}\t Recall: {:.4f}\t precision: {:.4f}\t'.format(epoch, phase, epoch_loss, epoch_acc, recall_value, precision_value))
            else:
                with open(log_txt, 'a+') as f:
                    f.write('{} Loss: {:.4f}\t Acc: {:.4f}\t Recall: {:.4f}\t precision: {:.4f}\n'.format(phase, epoch_loss, epoch_acc, recall_value, precision_value))

            state = {
                'net': model.state_dict(),
                'acc': epoch_acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(checkpoint_dir, 'resnet18' + '.pth.tar'))

            is_best = False
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(checkpoint_dir, 'resnet18' + '.pth.tar'),
                                os.path.join(checkpoint_dir, 'resnet18' + '_best.pth.tar'))
            
            precision.reset()
            recall.reset()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    with open(log_txt, 'a+') as f:
        f.write('Training complete in {:.0f}m\t {:.0f}s\n Best test Acc: {:4f}\n'.format(time_elapsed // 60, time_elapsed % 60, best_acc))

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    train_model(model, criterion, optimizer, exp_lr_scheduler)
