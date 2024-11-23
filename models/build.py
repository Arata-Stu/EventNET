from omegaconf import DictConfig

from .backbone.darknet import CSPDarknet
from .backbone.rvt import RVT
from .backbone.sast import SAST
from .neck.pafpn import PAFPN
from .head.yolox_head import YOLOXHead
def build_backbone(backbone_config: DictConfig):
    name = backbone_config.name
    if name == 'darknet':
        print('darknet')
        backbone = CSPDarknet(dep_mul=backbone_config.depth,
                      wid_mul=backbone_config.width,
                      input_dim=backbone_config.input_dim,
                      out_features=backbone_config.out_features,
                      depthwise=backbone_config.depthwise,
                      act=backbone_config.act)
    elif name == 'RVT':
        print('RVT')
        backbone = RVT(mdl_config=backbone_config)
    elif name=='SAST':
        print('SAST')
        backbone = SAST(mdl_config=backbone_config)
    else:
        NotImplementedError
    
    return backbone

def build_neck(neck_config: DictConfig, in_channels):
    name = neck_config.name
    if  name == 'pafpn':
        print('PAFPN')

        neck = PAFPN(
            depth=neck_config.depth,  
            in_stages=neck_config.in_stages,  
            in_channels=in_channels,  
            depthwise=neck_config.depthwise,  
            act=neck_config.act  
        )
    else:
        NotImplementedError
    
    return neck

def build_head(head_config: DictConfig, in_channels, strides):
    name = head_config.name
    if name == 'yolox':
        print('YOLOX-Head')
        head = YOLOXHead(
            num_classes=head_config.num_classes,   
            strides=strides,  
            in_channels=in_channels,  
            act=head_config.act, 
            depthwise=head_config.depthwise  
        )
        head.initialize_biases(prior_prob=0.01)
    else:
        NotImplementedError

    return head