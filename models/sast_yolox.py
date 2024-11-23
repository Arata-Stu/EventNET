import torch.nn as nn
from omegaconf import DictConfig
from .build import build_backbone, build_neck, build_head


class SASTYOLOX(nn.Module):
    def __init__(self, model_config: DictConfig):
        super().__init__()

        backbone_config = model_config.backbone
        neck_config = model_config.neck
        head_config = model_config.head

        self.backbone = build_backbone(backbone_config=backbone_config)

        in_channels = self.backbone.get_stage_dims(neck_config.in_stages)
        self.neck = build_neck(neck_config=neck_config, in_channels=in_channels)
        print('neck input channels', in_channels)
        strides = self.backbone.get_strides(neck_config.in_stages)
        print('head strides', strides)
        self.head = build_head(head_config=head_config, in_channels=in_channels, strides=strides)

    def forward_backbone(self,
                         x,
                         previous_states = None,
                         token_mask = None):
    
        backbone_features, states = self.backbone(x, previous_states, token_mask)
        return backbone_features, states
    
    def forward_detect(self,
                       backbone_features,
                       targets = None):
        
        neck_features = self.neck(backbone_features)
        if self.training:
            assert targets is not None    
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                neck_features, targets)
            
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            outputs = outputs["total_loss"]
            return outputs
        else:
            outputs = self.head(neck_features)
        
            return outputs

    def forward(self, 
                x, 
                previous_states = None,
                targets = None,
                token_mask = None):
        backbone_features, states = self.forward_backbone(x, previous_states, token_mask=token_mask)
        
        outputs = self.forward_detect(backbone_features, targets)
        return outputs, states