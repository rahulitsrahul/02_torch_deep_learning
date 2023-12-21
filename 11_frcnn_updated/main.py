import numpy as np
import pandas as pd
import os

import random
from PIL import Image, ImageDraw
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import json
import os

from parse_annoation_file import *
from generate_dataset import *
from model_frcnn_resnet50 import *
from dataloader_handler import *
from train_model import *

if __name__=='__main__':
    imgs_path = r"data"
    annotation_file = 'train_tubes_annotation.json'
    f = os.path.join(imgs_path, annotation_file)
    im_ext = '.jpg'
    train_imgs_folder = os.path.join(imgs_path, 'train')
    val_imgs_folder = os.path.join(imgs_path, 'val')
    
    generate_df = parse_annotation_file(f, im_ext)
    
    df = generate_df.df
    unique_imgs = df.image_id.unique()
    
    
    train_inds, val_inds = train_test_split(range(unique_imgs.shape[0]), test_size=0.1)
    num_data_copies = 2 # Copies n times the dataset and fed to the model
    train_inds = [i for i in range(len(unique_imgs))] * num_data_copies
    
    batch_size = 1
    shuffle_data = True
    train_dataloader = dataloader_handler(df, unique_imgs, train_inds, train_imgs_folder, im_ext, batch_size, shuffle_data)
    val_dataloader = dataloader_handler(df, unique_imgs, val_inds, train_imgs_folder, im_ext, batch_size, shuffle_data)
    
    
    num_classes = 1 + len(df['bbox_id'].unique())
    model = model_frcnn_resnet50(num_classes)
    
    lr=0.001
    momentum=0.9
    weight_decay=0.0005
    num_epochs = 100
    
    train_model(model, train_dataloader, lr, momentum, weight_decay, num_epochs, val_imgs_folder)
    
    