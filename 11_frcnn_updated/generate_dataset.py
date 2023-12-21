import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision import transforms as T
import albumentations as A
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

class generate_dataset(Dataset):
    def __init__(self, df, unique_imgs, indices, imgs_folder, im_ext):
        self.df = df
        self.unique_imgs = unique_imgs
        self.indices = indices
        self.imgs_folder = imgs_folder
        self.im_ext = im_ext
        self.transform = self.update_transform()
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image_name = self.unique_imgs[self.indices[idx]]
        boxes = self.df[self.df.image_id == image_name].values[:, 1:-1].astype('float')
        boxes=boxes.astype(np.int64).tolist()
        bbox_labels = self.df[self.df.image_id == image_name].values[:, -1].astype('int64')
        img = Image.open(os.path.join(self.imgs_folder, (image_name + self.im_ext))).convert("RGB")
        labels = torch.from_numpy(bbox_labels)
        
        labeled_boxes = []
        for i, box in enumerate(boxes):
            labeled_box = box + [bbox_labels[i]]  # Append label to the end of each bounding box
            labeled_boxes.append(labeled_box)
        
        transformed = self.transform(
            image=np.array(img),
            bboxes=labeled_boxes  # Pass the updated bounding boxes with labels
        )
        
        # transformed = self.transform(image=np.array(img),
        #                              bboxes=boxes,
        #                               class_labels=bbox_labels,
        #                               bboxes_lables=bbox_labels,
        #                              )
        
        img_transformed = transformed['image']
        boxes_transformed = transformed['bboxes']
        boxes_transformed = np.array(boxes_transformed).astype(np.int64).tolist()
        boxes_transformed_no_labels = []
        for box in boxes_transformed:
            boxes_transformed_no_labels.append(box[0:-1])
        # labels=transformed['class_labels']
        
        show_transform=False
        if show_transform:
            self.visualize_transform(img_transformed, boxes_transformed_no_labels)
        
        target = {}
        target['boxes'] = torch.tensor(boxes_transformed_no_labels)
        target['labels'] = labels
        return T.ToTensor()(img_transformed), target
    
    def update_transform(self):
        return A.Compose([
            A.Sequential([
                A.RandomRotate90(p=0.5),
                
                # A.Rotate(limit=30, mask_value=(255,255,255), p=0.5),
                # A.ShiftScaleRotate(shift_limit=0.2,
                #                    scale_limit=0,
                #                    rotate_limit=0,
                #                    border_mode=cv2.BORDER_CONSTANT,
                #                    value=(100,100,100),
                #                    p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=0.5),
                ], p=0.5)
            ],
            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.7, 
                                      # label_fields=['bbox_labels']
                                     )
            
            )
    
    def visualize_transform(self, img, boxes):
        img = Image.fromarray(img)
        img_bb = ImageDraw.Draw(img)
        
        for box in boxes:
            # Extracting coordinates from the NumPy array
            x, y, w, h = box
            
            # Calculate the coordinates of the rectangle
            top_left = (x, y)
            bottom_right = (w, h)
            
            # Draw a rectangle on the image
            img_bb.rectangle(top_left + bottom_right, outline='red', width=5)  # You can change the outline color if needed
        
        # Save the modified image or display it
        plt.imshow(img)
        plt.show()
        