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

import albumentations as A

from model_frcnn_resnet50 import *

import json
import os

class model_inference(object):
    def __init__(self, model, pred_thresh=0.8):
        self.model = model
        self.pred_thresh = pred_thresh
        self.out_bbox_thresh = None
        self.out_labels_thresh = None
        
    def infer_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        im = self.transform_img(img)
        
        # Model Prediction
        output = model([im])
        
        out_bbox = output[0]['boxes'].cpu().detach().numpy().astype(np.int64)
        out_scores = output[0]['scores'].cpu().detach().numpy()
        out_labels = output[0]['labels'].cpu().detach().numpy().astype(np.int64)
        
        out_scores_bool = out_scores > self.pred_thresh
        self.out_bbox_thresh = out_bbox[out_scores_bool]
        self.out_labels_thresh = out_labels[out_scores_bool]
        
        self.visualize_predictions(img)
    
    # Infer on input image (get bbox, class labels), then scale the image and transform bboxes according to the scale
    def infer_img_alb_scale(self, img_path, alb_scale):
        img = Image.open(img_path).convert('RGB')
        im = self.transform_img(img)
        
        # Model Prediction
        output = model([im])
        
        out_bbox = output[0]['boxes'].cpu().detach().numpy().astype(np.int64)
        out_scores = output[0]['scores'].cpu().detach().numpy()
        out_labels = output[0]['labels'].cpu().detach().numpy().astype(np.int64)
        
        out_scores_bool = out_scores > self.pred_thresh
        out_bbox = out_bbox[out_scores_bool]
        out_labels = out_labels[out_scores_bool]
        
        alb_transform = self.get_alb_transform_scale(size_array=alb_scale)
        im_transform = alb_transform(image=np.array(img), bboxes=out_bbox.tolist(), category_ids=out_labels.tolist())
        im_transformed = im_transform['image']
        bbox_transformed = np.array(im_transform['bboxes']).astype(np.int64).tolist()
        out_labels_transformed = np.array(im_transform['category_ids']).astype(np.int64).tolist()
        
        img_transformed = Image.fromarray(np.array(im_transformed))
        self.out_bbox_thresh = bbox_transformed
        self.out_labels_thresh = out_labels_transformed
        
        self.visualize_predictions(img_transformed)
        
    def infer_img_alb_rotate(self, img_path, alb_angle): # rotate CCW
        img = Image.open(img_path).convert('RGB')
        img_size = img.size
        img = img.rotate(angle=alb_angle, expand=True)
        im = self.transform_img(img)
        
        # Model Prediction
        output = model([im])
        
        out_bbox = output[0]['boxes'].cpu().detach().numpy().astype(np.int64)
        out_scores = output[0]['scores'].cpu().detach().numpy()
        out_labels = output[0]['labels'].cpu().detach().numpy().astype(np.int64)
        
        out_scores_bool = out_scores > self.pred_thresh
        out_bbox = out_bbox[out_scores_bool]
        out_labels = out_labels[out_scores_bool]
        
        alb_transform = self.get_alb_transform_rotate(alb_angle=-alb_angle, img_size=img_size)
        im_transform = alb_transform(image=np.array(img), bboxes=out_bbox.tolist(), category_ids=out_labels.tolist())
        im_transformed = im_transform['image']
        bbox_transformed = np.array(im_transform['bboxes']).astype(np.int64).tolist()
        out_labels_transformed = np.array(im_transform['category_ids']).astype(np.int64).tolist()
        
        img_transformed = Image.fromarray(np.array(im_transformed))
        self.out_bbox_thresh = bbox_transformed
        self.out_labels_thresh = out_labels_transformed
        
        self.visualize_predictions(img_transformed)
    
    def transform_img(self, img):
        # Transfrom image
        im = T.ToTensor()(img)
        return im
    
    def get_alb_transform_scale(self, size_array):
        transform = A.Compose([
            A.Resize(width=size_array[0], height=size_array[1])
            ], A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        return transform
    
    def get_alb_transform_rotate(self, alb_angle, img_size):
        transform = A.Compose([
            # Resize the input image to 'square' dimensions to avoid crop druing rotation
            A.Resize(width=max(img_size), height=max(img_size)),
            A.Rotate(limit=(alb_angle, alb_angle), p=1), # Rotate to the specified angle
            # Resize the rotated image to the original dimensions
            A.Resize(width=img_size[0], height=img_size[1]),
            ], A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        return transform
    
    def visualize_predictions(self, img):
        if(len(self.out_bbox_thresh) > 0):
            out_lables = self.out_labels_thresh
            img_bb = ImageDraw.Draw(img)
            
            class_color_list = ['black', 'red', 'green', 'blue', 'yellow']
            for out_bbox, label in zip(self.out_bbox_thresh, self.out_labels_thresh):
                bboxes = out_bbox
            
                # Extracting coordinates from the NumPy array
                xmin, ymin, xmax, ymax = bboxes
                
                # Calculate the coordinates of the rectangle
                top_left = (xmin, ymin)
                bottom_right = (xmax, ymax)
                
                # Draw a rectangle on the image
                img_bb.rectangle([top_left, bottom_right], outline=class_color_list[label], width=5)  # You can change the outline color if needed
            
        # Save the modified image or display it
        plt.imshow(img)
        plt.show()

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    WEIGHTS_FILE = r'frcnn_model_epoch_30.pth'
    
    num_classes = 1 + 2
    model = model_frcnn_resnet50(num_classes)
    # load the trained weights
    model.load_state_dict(torch.load(os.path.join('model_weights', WEIGHTS_FILE)))
    model.eval()
    
    model_infer = model_inference(model=model, pred_thresh=0.9)
    
    ##----------------------------------TEST on Unseen DATA----------------------------##
    image_folder = r"data\test"
    img_name = 'IMG_4897_JPG_jpg.rf.85c4e1cc7d83725f079cd20968b70635.jpg'
    img_path = os.path.join(image_folder, img_name)
    
    model_infer.infer_img(img_path=img_path)
    # model_infer.infer_img_alb_scale(img_path=img_path, alb_scale=[300, 150])
    model_infer.infer_img_alb_rotate(img_path=img_path, alb_angle=180)
    # torch.cuda.empty_cache() # Free the cache memory of cuda
    
