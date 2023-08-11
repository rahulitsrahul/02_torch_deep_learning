import numpy as np
import pandas as pd
import os

import random
from PIL import Image, ImageDraw
from collections import Counterimport 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision import transfroms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import json
import os

# Create df table
imgs_path = "data_img"
f = os.path.join(imgs_path, 'annotations.json')

file = open(f, encoding="utf-8")
json_data = json.load(file)
file.close()
data = json_data['_via_img_metadata']

df = pd.DataFrame(columns=["image_id", "x1", "y1", "x2", "y2", "bbox_id"])
index=0

elements = list(data.keys())
for el in elements:
    cur_el = data[el]
    file_name= cur_el['filename']
    regions = cur_el['regions']
    
    for region in regions:
        label = region['region_attributes']['label']
        x = region['shape_attributes']['x']
        y = region['shape_attributes']['y']
        width = region['shape_attributes']['width']
        height = region['shape_attributes']['height']
        
        if(label == 'sentence'):
            label_value = 100
            id_val = 4
        elif(label == 'topic'):
            label_value = 64
            id_val = 3
        elif(label =='header'):
            label_value = 128
            id_val = 2
        elif(label == 'content'):
            label_value = 255
            id_val = 1
            
        image_id = file_name.split('.jpg')[0]
        x1 = x
        y1 = y
        x2 = x1 + width
        y2 = y1 + height
        
        df.loc[index] = [image_id, x1, y1, x2, y2, id_val]
        index += 1
        
unique_imgs = df.image_id.unique()

class CustDat(torch.utils.data.Dataset):
    def __init__(self, df, unique_imgs, indices):
        self.df = df
        self.unique_imgs = unique_imgs
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image_name = self.unique_imgs[self.indices[idx]]
        boxes = self.df[self.df.image_id == image_name].values[:, 1:-1].astype('float')
        bbox_labels = self.df[self.df.image_id == image_name].values[:, -1].astype('int64')
        img = Image.open('data_img/train/' + image_name + '.jpg').convert("RGB")
        labels = torch.from_numpy(bbox_labels)
        target = {}
        target['boxes'] = torch.tensor(boxes)
        target['labels'] = labels
        return T.ToTensor()(img), target
    
    train_inds, val_inds = train_test_split(range(unique_imgs.shape[0]), test_size=0.1)
    
def custom_collate(data):
    return data

train_dl = torch.utils.data.DataLoader(CustDat(df, unique_imgs, train_inds),
                                        batch_size=4,
                                        shuffle=True,
                                        collate_fn = custom_collate,
                                        pin_memory = True if torch.cuda.is_available() else False
                                        )

val_dl = torch.utils.data.DataLoader(CustDat(df, unique_imgs, train_inds),
                                        batch_size=4,
                                        shuffle=True,
                                        collate_fn = custom_collate,
                                        pin_memory = True if torch.cuda.is_available() else False
                                        )

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 1 + len(df['bbox_id'].unique())
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
num_epochs = 100

model.to(device)

print("######---------Training Started----------#####")
for epochs in range(num_epochs):
    epoch_loss=0
    for data in train_dl:
        imgs = []
        targets = []
        for d in data:
            imgs.append(d[0].to(device))
            targ={}
            targ['boxes'] = d[1]['boxes'].to(device)
            targ['labels'] = d[1]['labels'].to(device)
            targets.append(targ)
            
        loss_dict = model(imgs, targets)
        loss = sum(v for v in loss_dict.values())
        epoch_loss += loss.cpu().detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    print(f'Epoch {epochs}, loss: {epoch_loss}')
    
model.eval()
data = iter(val_dl).__next__()
img = data[0][0]
boxes = data[0][1]['boxes']
lables = data[0][1]['labels']

output = model([img.to(device)])
print(output)

out_bbox = output[0]['boxes']
out_scores = output[0]['scores']

keep = torchvision.ops.nms(out_bbox, out_scores, 0.35)
print(out_bbox.shape), print(keep.shape)

pred_bbox = out_bbox[keep]

im = (img.permute(1,2,0).cpu().detach().numpy() * 255).astype('uint8')
vsample = Image.fromarray(im)
draw = ImageDraw.Draw(vsample)
# for box in boxes:
for box in pred_bbox:
    draw.rectangle(list(box), fill=None, outline='red')
vsample.shaoe()

for box in boxes:
    draw.rectangle(list(box), fill=None, outline='red')
vsample.shaoe()


##----------------------------------TEST on Unseen DATA----------------------------##
image_name = "Device type mismatch page 0"
img = Image.open('data_img/test/' + image_name + '.jpg').convert('RGB')
im = T.ToTensor()(img)


output = model([im.to(device)])
print(output)

out_bbox = output[0]['boxes']
out_scores = output[0]['scores']
out_labels = output[0]['labels']

#---------------------------------------------
pred_threshold = 0.81
out_scores_bool = out_scores > pred_threshold
out_bbox_thresh = out_bbox[out_scores_bool]
out_labels_thresh = out_labels[out_scores_bool]

im = (im.permute(1,2,0).cpu().detach().numpy() * 255).astype('uint8')
vsample = Image.fromarray(im)
draw = ImageDraw.Draw(vsample)

for box, label in zip(out_bbox_thresh, out_labels_thresh):
    if label == 1:
        draw.rectangle(list(box), fill=None, outline='red')
    elif label ==2:
        draw.rectangle(list(box), fill=None, outline='blue')
    elif label ==3:
        draw.rectangle(list(box), fill=None, outline='green')
    elif label ==2:
        draw.rectangle(list(box), fill=None, outline='magenta')
vsample.show()

save_model=True
if save_model:
    torch.save(model.state_dict(), "fastrcnn_resnet50_fpn_3.pth")
        
    

