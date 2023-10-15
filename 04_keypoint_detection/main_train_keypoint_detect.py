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
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

import json
import os


os.chdir("../")

# Create df table
im_ext = ".jpg"

imgs_path = r"04_keypoint_detection\data_img"
f = os.path.join(imgs_path, 'train', 'annotations')

img_annotatations = os.listdir(f)

# Initialize empty lists to store data
image_names = []
bboxes_list = []
keypoints_list = []

# Iterate through all the JSON files in the directory and load their content
for filename in os.listdir(f):
    if filename.endswith(".json"):
        json_path = os.path.join(imgs_path, 'train', 'annotations', filename)
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        bboxes_list.extend(json_data["bboxes"])
        keypoints_list.extend(json_data["keypoints"])
        file_id = filename.split(".json")[0]
        image_names.extend([file_id] * len(json_data["bboxes"]))  # Associate image name with each bbox/keypoint


# Create a DataFrame with the loaded data
data = {
    "image_name": image_names,
    "bboxes": bboxes_list,
    "keypoints": keypoints_list
}
df_init = pd.DataFrame(data)

# Print the DataFrame to verify
print(df_init.iloc[0])

df = pd.DataFrame(columns=["image_id", "x1", "y1", "x2", "y2", "bbox_id", "x", "y", "visibility"])
index=0
for i in range(len(df_init)):
    df.loc[i] = [
        df_init.iloc[i]['image_name'],
        df_init.iloc[i]['bboxes'][0],
        df_init.iloc[i]['bboxes'][1],
        df_init.iloc[i]['bboxes'][2],
        df_init.iloc[i]['bboxes'][3],
        1, # bbox id
        df_init.iloc[i]['keypoints'][0][0],
        df_init.iloc[i]['keypoints'][0][1],
        df_init.iloc[i]['keypoints'][0][2],
        ]
    
unique_imgs = df_init.image_name.unique()

class CustDat(torch.utils.data.Dataset):
    def __init__(self, df, unique_imgs, indices):
        self.df = df
        self.unique_imgs = unique_imgs
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image_name = self.unique_imgs[self.indices[idx]]
        boxes = self.df[self.df.image_id == image_name].values[:,1:5].astype('float')
        bbox_labels = self.df[self.df.image_id == image_name].values[:,5].astype('int64')
        key_points = self.df[self.df.image_id == image_name].values[:,6:10].astype('float32')
        im_name=image_name + im_ext
        im_loc = os.path.join(imgs_path, 'train', 'images', im_name)
        img = Image.open(im_loc).convert("RGB")
        labels = torch.from_numpy(bbox_labels)
        target = {}
        target['boxes'] = torch.tensor(boxes)
        target['labels'] = labels
        target['keypoints'] = torch.tensor(key_points)
        return T.ToTensor()(img), target
    
train_inds, val_inds = train_test_split(range(unique_imgs.shape[0]), test_size=0.1)
    
def custom_collate(data):
    return data


train_dl = torch.utils.data.DataLoader(CustDat(df, unique_imgs, train_inds),
                                        batch_size=1,
                                        shuffle=True,
                                        collate_fn = custom_collate,
                                        pin_memory = True if torch.cuda.is_available() else False
                                        )

val_dl = torch.utils.data.DataLoader(CustDat(df, unique_imgs, train_inds),
                                        batch_size=1,
                                        shuffle=True,
                                        collate_fn = custom_collate,
                                        pin_memory = True if torch.cuda.is_available() else False
                                        )

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   )
num_classes = 1 + 1
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = KeypointRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
num_epochs = 10

model.to(device)

print("######---------Training Started----------#####")
for epochs in range(num_epochs):
    epoch_loss=0
    for data in train_dl:
        imgs = [d[0].to(device) for d in data]
        targets = [{
            'boxes': d[1]['boxes'].to(device),
            'labels': d[1]['labels'].to(device),
            'keypoints': d[1]['keypoints'].to(device)
        } for d in data]
        
        optimizer.zero_grad()
        loss_dict = model(imgs, targets)
        loss = sum(v for v in loss_dict.values())
        epoch_loss += loss.cpu().detach().numpy()
        
        loss.backward()
        optimizer.step() 
    print(f'Epoch {epochs}, loss: {epoch_loss}')
    
    
