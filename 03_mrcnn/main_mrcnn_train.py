import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
import matplotlib.pyplot as plt
import copy

os.chdir("../")

batchSize = 2
imageSize=[600, 600]
num_of_classes = 1 + 3 # Background + other classes
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mask_color_code = {"Umpire": 83, "Batsman": 133, "Bowler": 42, "Fielder": 219, "Keeper":192 }
mask_color_code_unique = np.array(list(mask_color_code.values()))

color_code_mask = {83:1, 133:2, 42:3, 219:3, 192:3} # mapping of class values into uniform lables, 1-> Umpire, 2-> India, 3-> Zimbabwe



train_imgs_path = r"03_mrcnn/data/train_imgs"
train_masks_path = r"03_mrcnn/data/train_masks"

train_imgs = os.listdir(train_imgs_path)
img_ids = [(img.split(".png"))[0] for img in train_imgs]
img_ids_temp = None

def loadData():
    batch_Imgs=[]
    batch_Data=[] # Load images and masks
    
    for i in range(batchSize):
        img_id = img_ids_temp.pop(0)
        img_path = os.path.join(train_imgs_path, img_id + ".png")
        mask_path = os.path.join(train_masks_path, img_id + ".png___fuse.png")
        print(f"Processing Img: {img_path}")
        img = cv2.imread(img_path)
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
        
        mask = cv2.imread(mask_path, 0)
        mask_val_unique = np.unique(mask)
        
        mask_val_img = np.intersect1d(mask_color_code_unique, mask_val_unique)
        
        # Get the instances of mask_val_img elements
        masks = []
        labels = []
        for msk_val in mask_val_img:
            mask_val_inst = (mask == msk_val).astype(np.uint8)
            num_labels , labels_img = cv2.connectedComponents(mask_val_inst)
            for label in range(1, num_labels+1):
                cur_obj_mask = (labels_img == label).astype(np.uint8)
                if(np.sum(cur_obj_mask) > 100):  # check if the total pixel count of the object is > 100 pixs
                    cur_obj_mask = cv2.resize(cur_obj_mask, imageSize, cv2.INTER_LINEAR)
                    masks.append(cur_obj_mask)
                    labels.append(color_code_mask[msk_val])
                    
        num_objs = len(labels)
        
        boxes = torch.zeros([num_objs, 4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data={}
        data["boxes"] = boxes
        data["labels"] = torch.tensor(labels, dtype=torch.int64)
        data["masks"] = masks
        batch_Imgs.append(img)
        batch_Data.append(data) # Load images and masks
        
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) # Load an instance segmentation mdoel pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features # Get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_of_classes) # replace the pre-trained head with a new one
model.to(device) # Move the model to the right device

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

num_epoch=1000
for i in range(num_epoch):
    img_ids_temp = copy.deepcopy(img_ids)
    while(not len(img_ids_temp) < batchSize):
        images, targets = loadData()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        print(i, "loss:", losses.item())
        
    if i%10 == 0: # Save the model after every 10 epochs
        print("--------------------Saving the model---------------------")
        torch.save(model.state_dict(), r"03_mrcnn/model/" + str(i) + r".torch")
        
            