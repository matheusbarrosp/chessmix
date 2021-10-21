import os
import sys
import time
import numpy as np
import argparse
import copy
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision.models.segmentation import fcn_resnet101, fcn_resnet50
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

import skimage.io as io

import albumentations as A
from albumentations.pytorch import ToTensorV2

import dataset

#python3 main.py --dataset ../vaihingen_1000/ --epochs 10 --batch_size 8 --num_workers 4 --n_classes 6 --lr 1e-4

parser = argparse.ArgumentParser(description='PyTorch FCN')
parser.add_argument('--dataset', type=str, default='vaihingen', help="Name of the dataset")
parser.add_argument('--network', type=str, default='resnet101', help="Name of the network (resnet101|resnet50)")
parser.add_argument('--exp_name', type=str, default='fcn-resnet101', help="Name of the experiment")
parser.add_argument('--fold', type=int, default=None, help="Fold number")
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
parser.add_argument('--iterations', type=int, default=None, help="Number of iterations")
parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
parser.add_argument('--num_workers', type=int, default=4, help="number of workers")
parser.add_argument('--n_classes', type=int, default=6, help="number of classes")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=5e-4, help="L2 penalty")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
parser.add_argument('--new_data_size', type=int, default=100, help="Number of new data inputs")
parser.add_argument('--resume', type=str, default='False', help="Resume training")
parser.add_argument('--resume_epoch', type=int, default=0, help="Starting epoch for resume training")
parser.add_argument('--resume_iou', type=float, default=0, help="Starting iou for resume training")
parser.add_argument('--resume_model', type=str, default='models/vaihingen_plus0/vaihingen_plus0_best.pth', help="Model for resume training")
parser.add_argument('--data_augmentation', type=str, default='False', help="Use or not data augmentation (i.e. data warping, not including ChessMix)")

args = parser.parse_args()

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

# Computing mean Intersection over Union (mIoU).
def evaluate(prds, labs):
    if 'grss' in args.dataset:
        int_sum = np.zeros(args.n_classes, dtype=np.float32)
        uni_sum = np.zeros(args.n_classes, dtype=np.float32)
        
        for prd, lab in zip(prds, labs):
            prd = prd.ravel()
            lab = lab.ravel()
            new_prd = prd[lab != -1]
            new_lab = lab[lab != -1]
            prd = new_prd
            lab = new_lab
            for c in range(args.n_classes):
                union = np.sum(np.logical_or(lab == c, prd == c))
                #union = np.sum(lab.ravel() == c)
                if union > 0:
                    uni_sum[c] += union
                    
                    intersection = np.sum(np.logical_and(lab == c, prd == c))
                    int_sum[c] += intersection

    elif 'coffee' in args.dataset:
        int_sum = np.zeros(1, dtype=np.float32)
        uni_sum = np.zeros(1, dtype=np.float32)
        
        for prd, lab in zip(prds, labs):
            union = np.sum(np.logical_or(lab.ravel() == 1, prd.ravel() == 1))
            if union > 0:
                uni_sum[0] += union
                
                intersection = np.sum(np.logical_and(lab.ravel() == 1, prd.ravel() == 1))
                int_sum[0] += intersection

    else:
        int_sum = np.zeros(args.n_classes, dtype=np.float32)
        uni_sum = np.zeros(args.n_classes, dtype=np.float32)
        
        for prd, lab in zip(prds, labs):
            
            for c in range(args.n_classes):
                union = np.sum(np.logical_or(lab.ravel() == c, prd.ravel() == c))
                #union = np.sum(lab.ravel() == c)
                if union > 0:
                    uni_sum[c] += union
                    
                    intersection = np.sum(np.logical_and(lab.ravel() == c, prd.ravel() == c))
                    int_sum[c] += intersection
        
    return int_sum, uni_sum


def train(train_loader, net, criterion, optimizer, epoch, ITERATIONS):
    tic = time.time()
    
    # Setting network for training mode.
    net.train()

    # Lists for losses and metrics.
    train_loss = []

    int_all = np.asarray(0, dtype=np.float32)
    uni_all = np.asarray(0, dtype=np.float32)

    # Iterating over batches.
    for i, batch_data in enumerate(train_loader):
        ITERATIONS += 1
        if args.iterations is not None and ITERATIONS > args.iterations:
            break
        #print('Train: epoch {}, iteration {}'.format(epoch, i))
        # Obtaining images and labels for batch.
        inps, labs, img_name = batch_data
        # Casting to cuda variables.
        inps = inps.to(device)
        labs = labs.to(device)

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding through network.
        outs = net(inps)
        outs = outs['out']
        # Computing loss.
        loss = criterion(outs.view(outs.size(0), outs.size(1), -1), labs.view(labs.size(0), -1))

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Obtaining predictions.
        prds = outs.data.max(1)[1].squeeze_(1).squeeze(0).cpu().numpy()
        
        # Appending metrics for epoch error calculation.
        int_sum, uni_sum = evaluate([prds], [labs.detach().squeeze(0).cpu().numpy()])

        int_all = int_all + int_sum
        uni_all = uni_all + uni_sum

        # Updating loss meter.
        train_loss.append(loss.data.item())

    toc = time.time()
    
    # Transforming list into numpy array.
    train_loss = np.asarray(train_loss)
    
    # Computing error metrics for whole epoch.
    iou = 0
    iou = np.divide(int_all, uni_all)
    # print(iou)

    # Printing training epoch loss and metrics.
    print('-------------------------------------------------------------------')
    print('[epoch %d], [iterations %d], [train loss %.4f +/- %.4f], [miou %.4f], [time %.4f]' % (
        epoch, ITERATIONS, train_loss.mean(), train_loss.std(), iou.mean(), (toc - tic)))
    print('-------------------------------------------------------------------')
    return ITERATIONS


def validate(val_loader, net, criterion, epoch):

    tic = time.time()
    
    # Setting network for evaluation mode.
    net.eval()

    # Lists for losses and metrics.
    val_loss = []

    int_all = np.asarray(0, dtype=np.float32)
    uni_all = np.asarray(0, dtype=np.float32)

    # Iterating over batches.
    for i, batch_data in enumerate(val_loader):
        #print('Validation: epoch {}, iteration {}'.format(epoch, i))
        # Obtaining images and labels for batch.
        inps, labs, img_name = batch_data

        # Casting to cuda variables.
        inps = inps.to(device)
        labs = labs.to(device)

        # Forwarding through network.
        outs = net(inps)
        outs = outs['out']
        # Computing loss.
        loss = criterion(outs.view(outs.size(0), outs.size(1), -1), labs.view(labs.size(0), -1))

        # Obtaining predictions.
        prds = outs.data.max(1)[1].squeeze_(1).squeeze(0).cpu().numpy()

        # Appending metrics for epoch error calculation.
        int_sum, uni_sum = evaluate([prds], [labs.detach().squeeze(0).cpu().numpy()])

        int_all = int_all + int_sum
        uni_all = uni_all + uni_sum

        # Updating loss meter.
        val_loss.append(loss.data.item())

    toc = time.time()
    
    # Transforming list into numpy array.
    val_loss = np.asarray(val_loss)
    
    # Computing error metrics for whole epoch.
    iou = 0
    iou = np.divide(int_all, uni_all)
    # print(iou)

    # Printing test epoch loss and metrics.
    print('-------------------------------------------------------------------')
    print('[epoch %d], [val loss %.4f +/- %.4f], [miou %.4f], [time %.4f]' % (
        epoch, val_loss.mean(), val_loss.std(), iou.mean(), (toc - tic)))
    print('-------------------------------------------------------------------')
    return iou.mean()


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)
if args.resume == 'False':
    if args.network == 'resnet101':
        model = fcn_resnet101(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, args.n_classes, kernel_size=(1,1), stride=(1,1))
        model.aux_classifier[4] = nn.Conv2d(256, args.n_classes, kernel_size=(1,1), stride=(1,1))
    else:
        model = fcn_resnet50(pretrained=False)
        pth = torch.load("fcn_resnet50_coco-1167a1af.pth")
        for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias", "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked", "aux_classifier.4.weight", "aux_classifier.4.bias"]:
            del pth[key]
        model.load_state_dict(pth)

        model.classifier[4] = nn.Conv2d(512, args.n_classes, kernel_size=(1,1), stride=(1,1))
else:
    model = torch.load(args.resume_model)

model = model.to(device)

if args.data_augmentation == 'True':
    A_transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            #A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.8),
            A.Perspective(p=0.8),
            #A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.5)])
else:
    A_transform = None

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

train_set = dataset.ListDataset(mode='train', dataset=args.dataset, fold=args.fold, new_data_size=args.new_data_size, transform=transform, A_transform=A_transform)
val_set = dataset.ListDataset(mode='val', dataset=args.dataset, fold=args.fold, new_data_size=0, transform=transform)

print('Train size: ', len(train_set))
print('Val size: ', len(val_set))

train_loader = DataLoader(train_set,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=True)
val_loader = DataLoader(val_set,
                        batch_size=1,
                        num_workers=args.num_workers,
                        shuffle=False)

criterion = nn.CrossEntropyLoss(ignore_index = -1).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.momentum, 0.999))

# Iterating over epochs.
check_mkdir('models/'+args.exp_name+'/')
best_model = copy.deepcopy(model)

if args.resume == 'False':
    best_iou = 0
    best_epoch = 0
    starting_epoch = 1
else:
    best_iou = args.resume_iou
    best_epoch = args.resume_epoch
    starting_epoch = args.resume_epoch+1
    
epoch = starting_epoch
ITERATIONS = 0
if args.iterations is None:
    for epoch in range(starting_epoch, args.epochs + 1):
        # Training function.
        train(train_loader, model, criterion, optimizer, epoch, ITERATIONS)

        # Computing test loss and metrics.
        iou = validate(val_loader, model, criterion, epoch)
        if iou > best_iou:
            best_model = copy.deepcopy(model)
            best_iou = iou
            best_epoch = epoch
            print('saving new best model (iou {})...'.format(iou))
            torch.save(best_model, 'models/'+args.exp_name+'/'+args.exp_name+'_best.pth')
        else:
            print('Previous best iou: {} at epoch {}'.format(best_iou, best_epoch))
else:
    while ITERATIONS < args.iterations:
        # Training function.
        ITERATIONS = train(train_loader, model, criterion, optimizer, epoch, ITERATIONS)

        # Computing test loss and metrics.
        iou = validate(val_loader, model, criterion, epoch)
        if iou > best_iou:
            best_model = copy.deepcopy(model)
            best_iou = iou
            best_epoch = epoch
            print('saving new best model (iou {})...'.format(iou))
            torch.save(best_model, 'models/'+args.exp_name+'/'+args.exp_name+'_best.pth')
        else:
            print('Previous best iou: {} at epoch {}'.format(best_iou, best_epoch))
        epoch += 1


check_mkdir('val_outputs/'+args.exp_name+'/')
model.eval()
# Iterating over batches.
for i, batch_data in enumerate(val_loader):

    # Obtaining images and labels for batch.
    inps, labs, img_name = batch_data

    # Casting to cuda variables.
    inps = inps.to(device)
    labs = labs.to(device)

    # Forwarding through network.
    outs = model(inps)
    outs = outs['out']
    # Obtaining predictions.
    prds = outs.data.max(1)[1].squeeze_(1).squeeze(0).cpu().numpy()
    label = labs.detach().squeeze(0).cpu().numpy()
    h, w = prds.shape
    if 'vaihingen' in args.dataset:
        new = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(h):
            for y in range(w):
                if prds[x][y] == 0:
                    new[x][y] = [255,255,255]
                    
                elif prds[x][y] == 1:
                    new[x][y] = [0,0,255]
                    
                elif prds[x][y] == 2:
                    new[x][y] = [0,255,255]
                    
                elif prds[x][y] == 3:
                    new[x][y] = [0,255,0]
                    
                elif prds[x][y] == 4:
                    new[x][y] = [255,255,0]
                    
                elif prds[x][y] == 5:
                    new[x][y] = [255,0,0]
                    
                else:
                    sys.exit('Invalid prediction')
        io.imsave('val_outputs/'+args.exp_name+'/'+img_name[0].replace('.tif', '.png'), new)
    
    elif 'grss' in args.dataset:
        new = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if label[i][j] == -1:
                    new[i][j] = [0,0,0]
                
                elif prds[i][j] == 0:
                    new[i][j] = [255,0,255]
                    
                elif prds[i][j] == 1:
                    new[i][j] = [0,255,0]
                    
                elif prds[i][j] == 2:
                    new[i][j] = [255,0,0]
                    
                elif prds[i][j] == 3:
                    new[i][j] = [0,255,255]
                    
                elif prds[i][j] == 4:
                    new[i][j] = [160,32,240]
                    
                elif prds[i][j] == 5:
                    new[i][j] = [46,139,87]

                elif prds[i][j] == 6:
                    new[i][j] = [255,255,0]
                    
                else:
                    sys.exit('Invalid prediction')
        io.imsave('val_outputs/'+args.exp_name+'/'+img_name[0].replace('.ppm', '.png'), new)

    elif 'coffee' in args.dataset:
        new = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if prds[i][j] == 0:
                    new[i][j] = [0,0,0]
                    
                elif prds[i][j] == 1:
                    new[i][j] = [255,255,255]
                    
                else:
                    sys.exit('Invalid prediction')
        io.imsave('val_outputs/'+args.exp_name+'/'+img_name[0].replace('.tif', '.png'), new)
