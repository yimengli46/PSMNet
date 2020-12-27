from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image

def test(imgL,imgR):
    model.eval()

    if args_cuda:
       imgL = imgL.cuda()
       imgR = imgR.cuda()     

    with torch.no_grad():
        disp = model(imgL,imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp

def gen_img_path(leftimg_path):
    # leftimg_path: 'leftImg8bit/val/frankfurt/frankfurt_000001_007407_leftImg8bit.png'
    b = leftimg_path.split('/') # ['leftImg8bit', 'val', 'frankfurt', 'frankfurt_000001_007407_leftImg8bit.png']
    c = b[3].split('_') # ['frankfurt', '000001', '007407', 'leftImg8bit.png']
    rightimg_path = '{}/{}/{}/{}_{}_{}_{}'.format('rightImg8bit', b[1], b[2], c[0], c[1], c[2], 'rightImg8bit.png')

    depth_path = '{}/{}/{}/{}_{}_{}_{}'.format('psmDepth', b[1], b[2], c[0], c[1], c[2], 'depth.png')

    folder_path = '{}/{}/{}'.format('psmDepth', b[1], b[2])
    return rightimg_path, depth_path, folder_path

args_KITTI = '2015'
args_loadmodel = 'trained_model/finetune_300.tar'
#args_leftimg = 'dataset/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_007407_leftImg8bit.png'
#args_rightimg = 'dataset/cityscapes/rightImg8bit/val/frankfurt/frankfurt_000001_007407_rightImg8bit.png'
args_model = 'stackhourglass'
args_maxdisp = 192
args_seed = 1
args_cuda = True

torch.manual_seed(args_seed)
if args_cuda:
    torch.cuda.manual_seed(args_seed)

if args_model == 'stackhourglass':
    model = stackhourglass(args_maxdisp)
elif args_model == 'basic':
    model = basic(args_maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args_loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args_loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))



normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
infer_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(**normal_mean_var)])    

mode = 'train'
img_list = np.load('dataset/cityscapes/{}_img_list.npy'.format(mode), allow_pickle=True)
#assert 1==2

for idx in range(len(img_list)):
    args_leftimg = img_list[idx]['rgb_path']
    print('idx = {}, args_leftimg = {}'.format(idx, args_leftimg))

    args_rightimg, depth_path, folder_path = gen_img_path(args_leftimg)

    args_leftimg = 'dataset/cityscapes/{}'.format(args_leftimg)
    args_rightimg = 'dataset/cityscapes/{}'.format(args_rightimg)
    depth_path = 'dataset/cityscapes/{}'.format(depth_path)
    folder_path = 'dataset/cityscapes/{}'.format(folder_path)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    imgL_o = Image.open(args_leftimg).convert('RGB')
    imgR_o = Image.open(args_rightimg).convert('RGB')

    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o) 

    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    start_time = time.time()
    pred_disp = test(imgL,imgR)
    print('time = %.2f' %(time.time() - start_time))


    if top_pad !=0 or right_pad != 0:
        img = pred_disp[top_pad:,:-right_pad]
    else:
        img = pred_disp

    baseline = 0.22
    focal = 2262
    depth = baseline * focal / img

    depth = (depth*256).astype('uint16')
    depth = Image.fromarray(depth)
    depth.save(depth_path)








