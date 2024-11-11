import torch.nn as nn
from omegaconf import DictConfig
from .build import build_backbone, build_neck, build_head


class YOLOX(nn.Module):
    def __init__(self, model_config: DictConfig):
        super().__init__()

        backbone_config = model_config.backbone
        neck_config = model_config.neck
        head_config = model_config.head

        self.backbone = build_backbone(backbone_config=backbone_config)

        in_channels = self.backbone.get_stage_dims(neck_config.in_stages)
        print('neck input channels', in_channels)
        self.neck = build_neck(neck_config=neck_config, in_channels=in_channels)
        strides = self.backbone.get_strides(neck_config.in_stages)
        print('head strides', strides)
        self.head = build_head(head_config=head_config, in_channels=in_channels, strides=strides)

    def forward(self, x, targets=None):
        backbone_features = self.backbone(x)
        neck_outs = self.neck(backbone_features)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                neck_outs, targets)
            
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            outputs = outputs["total_loss"]
        else:
            outputs = self.head(neck_outs)
      
        return outputs