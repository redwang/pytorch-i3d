import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-clip_batch_size', type=int, help='batch size of every forward')
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset


def run(max_steps=64e3, mode='rgb', root='/ssd2/charades/Charades_v1_rgb', clip_batch_size=8, load_model='', save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()
    i3d.eval() # Set model to evaluate mode
    clip_length = 16

    for phase in ['train', 'val']:
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels, name = data
            if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
                continue
            
            inputs = inputs.squeeze()
            c,t,h,w = inputs.shape
            padding_tensor = torch.zeros(c,clip_length,h,w)
            inputs = torch.cat((padding_tensor, inputs, padding_tensor))
            features = []
            batch_inputs = []
            for i in range(clip_length, t+clip_length):
                if len(batch_inputs) == clip_batch_size:
                    batch_inputs = torch.stack(batch_inputs)
                    ip = batch_inputs.cuda()
                    with torch.no_grad():
                        feat = i3d.extract_features(ip)
                    features.append(
                        feat.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
                    batch_inputs = []
                else:
                    frame_input = batch_inputs[:,i-clip_length/2:i+clip_length/2]
                    batch_inputs = batch_inputs.append(frame_input)

            if len(batch_inputs) > 0:
                batch_inputs = torch.stack(batch_inputs)
                ip = batch_inputs.cuda()
                with torch.no_grad():
                    feat = i3d.extract_features(ip)
                features.append(
                    feat.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
            np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, clip_batch_size=args.clip_batch_size, load_model=args.load_model, save_dir=args.save_dir)
