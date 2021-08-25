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
import skimage.io as io
import seaborn as sns
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

import dataset
import evaluation

#python3 main.py --dataset ../vaihingen_1000/ --epochs 10 --batch_size 8 --num_workers 4 --n_classes 6 --lr 1e-4

parser = argparse.ArgumentParser(description='PyTorch FCN')
parser.add_argument('--dataset', type=str, default='vaihingen_1000', help="Name of the dataset")
parser.add_argument('--network', type=str, default='resnet101', help="Name of the network (resnet101|resnet50)")
parser.add_argument('--exp_name', type=str, default='vaihingen_plus0', help="Name of the experiment")
parser.add_argument('--model_path', type=str, default='models/vaihingen_plus0/vaihingen_plus0_best.pth', help="Path to the pretrained model")
parser.add_argument('--fold', type=int, default=None, help="Fold number")
parser.add_argument('--n_classes', type=int, default=6, help="number of classes")
parser.add_argument('--save_imgs', type=str, default='True', help="Save or not the resulting thematic maps (True|False)")

args = parser.parse_args()

def normalize_rows(array):
    sum = array.sum(axis=1)
    new = np.zeros(array.shape)
    for i in range(array.shape[0]):
        new[i] = array[i]/sum[i]
    return new


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


def test(test_loader, net):

    tic = time.time()
    
    # Setting network for evaluation mode.
    net.eval()

    # Lists for metrics.
    #int_all = np.asarray(args.n_classes, dtype=np.float32)
    #uni_all = np.asarray(args.n_classes, dtype=np.float32)
    int_all = np.asarray(0, dtype=np.float32)
    uni_all = np.asarray(0, dtype=np.float32)

    check_mkdir('test_outputs/'+args.exp_name+'/')
    gts_all = []
    preds_all = []
    # Iterating over batches.
    for i, batch_data in enumerate(test_loader):
        #print('Validation: epoch {}, iteration {}'.format(epoch, i))
        # Obtaining images and labels for batch.
        inps, labs, img_name = batch_data

        # Casting to cuda variables.
        inps = inps.to(device)
        labs = labs.to(device)

        # Forwarding through network.
        outs = net(inps)
        outs = outs['out']

        # Obtaining predictions.
        prds = outs.data.max(1)[1].squeeze_(1).squeeze(0).cpu().numpy()
        preds_all.append(prds)
        gts_all.append(labs.detach().squeeze(0).cpu().numpy())
        label = labs.detach().squeeze(0).cpu().numpy()
        # Appending metrics for epoch error calculation.
        int_sum, uni_sum = evaluate([prds], [labs.detach().squeeze(0).cpu().numpy()])

        int_all = int_all + int_sum
        uni_all = uni_all + uni_sum
        
        if args.save_imgs == 'True':
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
                io.imsave('test_outputs/'+args.exp_name+'/'+img_name[0].replace('.tif', '.png'), new)
            
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
                io.imsave('test_outputs/'+args.exp_name+'/'+img_name[0].replace('.ppm', '.png'), new)

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
                io.imsave('test_outputs/'+args.exp_name+'/'+img_name[0].replace('.tif', '.png'), new)

    toc = time.time()
    
    # Computing error metrics for whole epoch.
    iou = 0
    iou = np.divide(int_all, uni_all)
    # print(iou)

    # Printing test epoch loss and metrics.
    print('-------------------------------------------------------------------')
    print('[miou %.4f], [time %.4f]' % (iou.mean(), (toc - tic)))
    print('-------------------------------------------------------------------')
    
    if 'grss' in args.dataset:
        new_preds = []
        new_gts = []
        for prd, lab in zip(preds_all, gts_all):
            prd = prd.ravel()
            lab = lab.ravel()
            new_prd = prd[lab != -1]
            new_lab = lab[lab != -1]       
            new_preds.append(new_prd)
            new_gts.append(new_lab)
        preds_all = new_preds
        gts_all = new_gts

    acc, acc_cls, mean_iou, iou, fwavacc, kappa = evaluation.evaluate(preds_all, gts_all, args.n_classes)
    print('[acc %.4f], [acc_cls %.4f], [iou %.4f], [fwavacc %.4f], [kappa %.4f]' % (acc, acc_cls, mean_iou, fwavacc, kappa))
    print('-------------------------------------------------------------------')

    #print(evaluation.confusion_matrix(preds_all, gts_all, args.n_classes))

    check_mkdir('heatmaps/'+args.exp_name+'/')
    if 'vaihingen' in args.dataset:
        y_labels = ['Impervious\nsurfaces', 'Building', 'Low\nvegetation', 'Tree', 'Car', 'Background']
        heatmap = normalize_rows(evaluation.confusion_matrix(preds_all, gts_all, args.n_classes))
        #heatmap = np.delete(heatmap, -1, axis=0)
        #fig = plt.figure(figsize=(5,5))
        fig = plt.figure(figsize=(6,6))
        ax = sns.heatmap(heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heatmaps/'+args.exp_name+'/'+args.exp_name+'_heatmap.png')
    elif 'grss' in args.dataset:
        y_labels = ['Road', 'Tree', 'Red roof', 'Grey roof', 'Concrete\nroof', 'Vegeta-\ntion', 'Bare soil']
        heatmap = normalize_rows(evaluation.confusion_matrix(preds_all, gts_all, args.n_classes))
        fig = plt.figure(figsize=(6,6))
        ax = sns.heatmap(heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heatmaps/'+args.exp_name+'/'+args.exp_name+'_heatmap.png')

    elif 'coffee' in args.dataset:
        y_labels = ['Non-coffee', 'Coffee']
        heatmap = normalize_rows(evaluation.confusion_matrix(preds_all, gts_all, args.n_classes))
        fig = plt.figure(figsize=(4,4))
        ax = sns.heatmap(heatmap, linewidth=0.5, cmap='Blues', annot=True, yticklabels=y_labels, xticklabels=False)
        fig.savefig('heatmaps/'+args.exp_name+'/'+args.exp_name+'_heatmap.png')


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)
#if args.network == 'resnet101':
#    model = fcn_resnet101(num_classes=args.n_classes)
#else:
#    model = fcn_resnet50(num_classes=args.n_classes)
model = torch.load(args.model_path)
#print(pth)
#model.load_state_dict(pth)
model = model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

test_set = dataset.ListDataset(mode='test', dataset=args.dataset, fold=args.fold, new_data_size=0, transform=transform)

print('Test size: ', len(test_set))

test_loader = DataLoader(test_set,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False)

# Iterating over epochs.
test(test_loader, model)

    
