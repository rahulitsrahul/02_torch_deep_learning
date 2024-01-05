import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Class Model FRCNN

class model_frcnn_resnet50(torch.nn.Module):
    def __init__(self, num_classes):
        super(model_frcnn_resnet50, self).__init__()
        
        # Load a pre-trained Faster R-CNN model --
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Modify the number of classes in the model
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        # Define anchor sizes and aspect ratios suitable for your dataset
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        
        # Get the region proposal network (RPN) and assign the anchor generator
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        self.model.rpn_anchor_generator = anchor_generator
        self.model.roi_heads.box_roi_pool = roi_pooler
    
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided.")
        
        if self.training:
            loss_dict = self.model(images, targets)
            return loss_dict
        else:
            outputs = self.model(images)
            return outputs