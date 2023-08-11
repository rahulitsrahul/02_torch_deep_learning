import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import os
import matplotlib.pyplot as plt

imageSize=[600, 600]
imgPath = "data/test_imgs/2022-08-24 (38).png"

num_of_classes = 1 + 4 # Background + other classes
mask_color_code = {1:"Umpire", 2:"Batsman", 3:"Ball", 4:"Wicket"}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) # Load an instance segmentation mdoel pre-trained on COCO
in_features = model.roi_headers.box_predictor.cls_score.in_features # Get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_of_classes) # replace the pre-trained head with a new one
model.load_state_dict(torch.load("model/350.torch"))
model.to(device) # Move the model to the right device
model.eval()

images = cv2.imread(imgPath)
images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
images = images.swapaxes(1, 3).swapaxes(2, 3)
images = list(image.to(device) for image in images)

with torch.no_grad():
    pred = model(images)
    
im = images[0].swapaxes(0,2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
im2 = im.copy()
for i in range(len(pred[0]["masks"])):
    msk=pred[0]["masks"][i,0].detach().cpu().numpy()
    scr=pred[0]["scores"][i].detach().cpu().numpy()
    label=pred[0]["labels"][i,0].detach().cpu().numpy()
    label=int(label)
    
    if scr>0.8:
        im2[:,:,0][msk>0.5] = random.randint(0, 255)
        im2[:,:,1][msk>0.5] = random.randint(0, 255)
        im2[:,:,2][msk>0.5] = random.randint(0, 255)
        
        print(label)
        plt.imshow((msk>0.5).astype(np.uint8))
        plt.show()
        
plt.imshow(im)
plt.imshow(im2)
plt.show()
    
    