import os, json, cv2, numpy as np, matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations as A # Library for augmentations

import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate
import json
import pandas as pd


##-------Extract from train json file
# os.chdir("../")

# Create df table
im_ext = ".jpg"

imgs_path = r"D:\02_my_learnings\01_python_repo\02_torch_deep_learning\04_keypoint_detection\data_img"
f = os.path.join(imgs_path, 'my_train_annotations.json')

file = open(f, encoding="utf-8")
json_data = json.load(file)
file.close()
# data = json_data['_via_img_metadata']
data = json_data

df = pd.DataFrame(columns=["image_id", "bbox", "bbox_labels", "keypoints"])
index=0

elements = list(data.keys())
for el in elements:
    cur_el = data[el]
    file_name= cur_el['filename']
    regions = cur_el['regions']
    image_id = file_name.split(im_ext)[0]
    
    for region in regions:
        label = region['region_attributes']['labels']
        
        if label=='kp_1':
            head_x = region['shape_attributes']['cx']
            head_y = region['shape_attributes']['cy']
            head = [head_x, head_y, 1]
            
        elif label=='kp_2':
            tail_x = region['shape_attributes']['cx']
            tail_y = region['shape_attributes']['cy']
            tail = [tail_x, tail_y, 1]
            
            
        elif label == 'bbox':
            x = region['shape_attributes']['x']
            y = region['shape_attributes']['y']
            width = region['shape_attributes']['width']
            height = region['shape_attributes']['height']
            
            x1 = x
            y1 = y
            x2 = x1 + width
            y2 = y1 + height
            
            bbox = [[x1, y1, x2, y2]]
    
    keypoints = [[head, tail]]
    bbox_labels = ['Glue tube']
    df.loc[index] = [image_id, bbox, bbox_labels, keypoints]
    index += 1
        
unique_imgs = df.image_id.unique()


def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )



class ClassDataset(Dataset):
    def __init__(self, imgs_path, df, unique_imgs, transform=None, demo=False):                
        
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.unique_imgs = unique_imgs
        self.df = df
        self.imgs_path = os.path.join(imgs_path, "my_train")
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_path, self.df.iloc[idx]['image_id'] + im_ext)

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)      
        
        bboxes_original = self.df.iloc[idx]['bbox']
        keypoints_original = self.df.iloc[idx]['keypoints']
        
        # All objects are glue tubes
        bboxes_labels_original = self.df.iloc[idx]['bbox_labels'] 
        
        # ##=---- for debug
        # if self.df.iloc[idx]['image_id'].startswith('IMG_4838'):
        #     print('debug')
        
        if self.transform:
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints            
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format            
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            
            # Apply augmentations
            transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']
            
            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1,2,2)).tolist()

            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened): # Iterating over objects
                obj_keypoints = []
                for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)
        
        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original        
        
        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are glue tubes
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)        
        img = F.to_tensor(img)
        
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64) # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)        
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.unique_imgs)
    
    
# KEYPOINTS_FOLDER_TRAIN = r'D:\02_my_learnings\01_python_repo\02_torch_deep_learning\04_keypoint_detection\data_img\train'
dataset = ClassDataset(imgs_path, df, unique_imgs, transform=train_transform(), demo=True)
# dataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=None, demo=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

iterator = iter(data_loader)
batch = next(iterator)

print("Original targets:\n", batch[3], "\n\n")
print("Transformed targets:\n", batch[1])    

keypoints_classes_ids2names = {0: 'Head', 1: 'Tail'}

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40,40))
        plt.imshow(image)

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)
        
image = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

keypoints = []
for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    keypoints.append([kp[:2] for kp in kps])

image_original = (batch[2][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

keypoints_original = []
for kps in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    keypoints_original.append([kp[:2] for kp in kps])

visualize(image, bboxes, keypoints, image_original, bboxes_original, keypoints_original)



def get_model(num_keypoints, weights_path=None):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

KEYPOINTS_FOLDER_TRAIN = r'D:\02_my_learnings\01_python_repo\02_torch_deep_learning\04_keypoint_detection\data_img\my_train'
KEYPOINTS_FOLDER_TEST = r'D:\02_my_learnings\01_python_repo\02_torch_deep_learning\04_keypoint_detection\data_img\test'

dataset_train = ClassDataset(imgs_path,df=df, unique_imgs=unique_imgs, transform=train_transform(), demo=False)
# dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=None, demo=False)
# dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=collate_fn)
# data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = get_model(num_keypoints = 2)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
num_epochs = 50

for epoch in range(num_epochs):
    print("-----Training started-----")
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
    lr_scheduler.step()
    print("-----Evaluating_model-----")
    # evaluate(model, data_loader_test, device)
    
# Save model weights after training
model_path = r"D:\02_my_learnings\01_python_repo\02_torch_deep_learning\04_keypoint_detection\model"
torch.save(model.state_dict(), os.path.join(model_path, 'keypointsrcnn_weights_test.pth'))


####-----------------------------------------------------------
# # Load Image from test folder and feed it to model
# iterator = iter(data_loader_test)
# images, targets = next(iterator)
# images = list(image.to(device) for image in images)

# with torch.no_grad():
#     model.to(device)
#     model.eval()
#     output = model(images)

# print("Predictions: \n", output)

# # ------Visualize Predictions
# image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
# scores = output[0]['scores'].detach().cpu().numpy()

# high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
# post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

# # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
# # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
# # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

# keypoints = []
# for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
#     keypoints.append([list(map(int, kp[:2])) for kp in kps])

# bboxes = []
# for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
#     bboxes.append(list(map(int, bbox.tolist())))
    
# visualize(image, bboxes, keypoints)

# #-------------------------------#
# #-----Test with a random_img
# im_path = 'D:/02_my_learnings/01_python_repo/02_torch_deep_learning/04_keypoint_detection/data_img/img_3.jpg'

# img = cv2.imread(im_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_test = F.to_tensor(img).to(device)

# im_test = [img_test]

# with torch.no_grad():
#     model.to(device)
#     model.eval()
#     output = model(im_test)

# print("Predictions: \n", output)

# # ------Visualize Predictions
# image = (im_test[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
# scores = output[0]['scores'].detach().cpu().numpy()

# high_scores_idxs = np.where(scores > 0.6)[0].tolist() # Indexes of boxes with scores > 0.7
# post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

# # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
# # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
# # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

# keypoints = []
# for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
#     keypoints.append([list(map(int, kp[:2])) for kp in kps])

# bboxes = []
# for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
#     bboxes.append(list(map(int, bbox.tolist())))
    
# visualize(image, bboxes, keypoints)

