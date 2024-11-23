import torch.nn as nn
from omegaconf import DictConfig
from .yolox import YOLOX
from .rvt_yolox import RVTYOLOX
from .sast_yolox import SASTYOLOX

def build_model(model_config: DictConfig):
    name = model_config.name
    if 'yolox' in name:
        print('yolox')
        model = YOLOX(model_config=model_config)
    elif 'rvt' in name:
        print('rvt')
        model = RVTYOLOX(model_config=model_config)
        pass
    elif 'sast' in name:
        print('sast')
        model = SASTYOLOX(model_config=model_config)
    else:
        NotImplementedError
    
    return model
