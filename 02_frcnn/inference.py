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

WEIGHTS_FILE = r'02_frcnn\model\fastrcnn_resnet50_fpn_3.pth'

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 1 + 3
# Get no. of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load the trained weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.eval()

x = model.to(device)

##----------------------------------TEST on Unseen DATA----------------------------##
image_name = r"02_frcnn\data_img\test\2022-08-24 (333).png"
img = Image.open(image_name).convert('RGB')
im = T.ToTensor()(img)


output = model([im.to(device)])
print(output)

out_bbox = output[0]['boxes']
out_scores = output[0]['scores']
out_labels = output[0]['labels']

#---------------------------------------------
pred_threshold = 0.6
out_scores_bool = out_scores > pred_threshold
out_bbox_thresh = out_bbox[out_scores_bool]
out_labels_thresh = out_labels[out_scores_bool]

im = (im.permute(1,2,0).cpu().detach().numpy() * 255).astype('uint8')
vsample = Image.fromarray(im)
draw = ImageDraw.Draw(vsample)
# ----------------------------------- OUTPUTS ----------------------------------#
# Show image with bounding box
for box, label in zip(out_bbox_thresh, out_labels_thresh):
    if label == 1:
        draw.rectangle(list(box), fill=None, outline='red')
    elif label ==2:
        draw.rectangle(list(box), fill=None, outline='blue')
    elif label ==3:
        draw.rectangle(list(box), fill=None, outline='green')

vsample.show()

