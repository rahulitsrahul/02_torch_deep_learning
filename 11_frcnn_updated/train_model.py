import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torchvision import transforms as T
import numpy as np
import os

class train_model(object):
    def __init__(self, model, train_dataloader, lr=0.001, momentum=0.9, weight_decay=0.0005, num_epochs=100, val_imgs_folder=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.val_imgs_folder = val_imgs_folder
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        
        self.optimizer = None
        self.get_optimizer()
        self.run_epochs()
        
    
    def get_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.lr, 
                                         momentum=self.momentum, 
                                         weight_decay=self.weight_decay,
                                         )
    
    
    def run_epochs(self):
        print("######---------Training Started----------#####")
        for epochs in range(self.num_epochs):
            epoch_loss=0
            for data in self.train_dataloader.dataloader_obj:
                imgs = []
                targets = []
                for d in data:
                    imgs.append(d[0].to(self.device))
                    targ={}
                    targ['boxes'] = d[1]['boxes'].to(self.device)
                    targ['labels'] = d[1]['labels'].to(self.device)
                    targets.append(targ)
                
                self.model.train()
                loss_dict = self.model(imgs, targets)
                loss = sum(v for v in loss_dict.values())
                epoch_loss += loss.cpu().detach().numpy()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.evaluate_model()
            print(f'Epoch {epochs}, loss: {epoch_loss}')
    
    def evaluate_model(self):
        self.model.eval()
        
        val_imgs_list = os.listdir(self.val_imgs_folder)
        val_imgs_list  = val_imgs_list
        for img_name in val_imgs_list:
            img_path = os.path.join(self.val_imgs_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            im = T.ToTensor()(img)


            output = self.model([im.to(self.device)])
            # print(output)

            out_bbox = output[0]['boxes']
            out_scores = output[0]['scores']
            out_labels = output[0]['labels']

            #---------------------------------------------
            pred_threshold = 0.90
            out_scores_bool = out_scores > pred_threshold
            out_bbox_thresh = out_bbox[out_scores_bool]
            out_labels_thresh = out_labels[out_scores_bool]
            
            if(len(out_bbox_thresh) > 0):
                out_lables = out_labels_thresh.cpu().detach().numpy().astype(np.int64)
                img_bb = ImageDraw.Draw(img)
                
                class_color_list = ['black', 'red', 'green', 'blue', 'yellow']
                for out_bbox, label in zip(out_bbox_thresh, out_labels_thresh):
                    bboxes = out_bbox.cpu().detach().numpy().astype(np.int64)
                
                    # Extracting coordinates from the NumPy array
                    x, y, w, h = bboxes
                    
                    # Calculate the coordinates of the rectangle
                    top_left = (x, y)
                    bottom_right = (w, h)
                    
                    # Draw a rectangle on the image
                    img_bb.rectangle([top_left, bottom_right], outline=class_color_list[label], width=5)  # You can change the outline color if needed
                
            # Save the modified image or display it
            plt.imshow(img)
            plt.show()
            
            
            
        
        
        