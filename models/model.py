import torch.nn as nn
from omegaconf import DictConfig
from .yolox import YOLOX
from .rvt_yolox import RVTYOLOX

def build_model(model_config: DictConfig):
    name = model_config.name
    if 'yolox' in name:
        print('yolox')
        model = YOLOX(model_config=model_config)
    elif 'rvt' in name:
        print('rvt')
        model = RVTYOLOX(model_config=model_config)
        pass
    else:
        NotImplementedError
    
    return model

# class DNNModel(nn.Module):
#     def __init__(self, model_config: DictConfig):
#         super().__init__()

#         self.model = build_model(model_config=model_config)
        
#     def forward(self, x, targets=None):
        
#         outputs = self.model(x, targets)
      
#         return outputs
    
# class RNNModel(nn.Module):
#     def __init__(self, model_config: DictConfig):
#         super().__init__()

#         self.model = build_model(model_config=model_config)

#     def forward_backbone(self, x, previous_states = None, token_mask = None):
#         backbone_features, states = self.model.forward_backbone(x=x,
#                                                             previous_states=previous_states,
#                                                             token_mask=token_mask)
        
#         return backbone_features, states
    
#     def forward_detect(self,
#                        backbone_features,
#                        targets = None):
        
#         neck_features = self.neck(backbone_features)
#         if self.training:
#             assert targets is not None    
#             loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
#                 neck_features, targets)
            
#             outputs = {
#                 "total_loss": loss,
#                 "iou_loss": iou_loss,
#                 "l1_loss": l1_loss,
#                 "conf_loss": conf_loss,
#                 "cls_loss": cls_loss,
#                 "num_fg": num_fg,
#             }
#             outputs = outputs["total_loss"]
#             return outputs
#         else:
#             outputs = self.head(neck_features)
        
#             return outputs
    
        
#     def forward(self, x, prev_states=None, token_mask=None, targets=None):
        
#         outputs = self.model(x, prev_states=prev_states, token_mask=token_mask, targets=targets)
      
#         return outputs