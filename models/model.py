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

class DNNModel(nn.Module):
    def __init__(self, model_config: DictConfig):
        super().__init__()

        self.model = build_model(model_config=model_config)
        
    def forward(self, x, targets=None):
        
        outputs = self.model(x, targets)
      
        return outputs
    
class RNNModel(nn.Module):
    def __init__(self, model_config: DictConfig):
        super().__init__()

        self.model = build_model(model_config=model_config)
        
    def forward(self, x, prev_states=None, token_mask=None, targets=None):
        
        outputs = self.model(x, prev_states=prev_states, token_mask=token_mask, targets=targets)
      
        return outputs