import os
from typing import Tuple

import math
from omegaconf import DictConfig, open_dict


def dynamically_modify_train_config(config: DictConfig):
    with open_dict(config):
        
        dataset_cfg = config.dataset
        dataset_name = dataset_cfg.name
        assert dataset_name in {'gen1', 'gen4', 'dsec'}

        dataset_cfg.data_dir = os.path.join('../../', config.data_dir)
        dataset_hw = dataset_cfg.orig_size

        mdl_cfg = config.model
        mdl_name = mdl_cfg.name

        #32の倍数になるようにheight widthを調整
        partition_split_32 = mdl_cfg.backbone.partition_split_32
        assert partition_split_32 in (1, 2, 4)
        multiple_of = 32 * partition_split_32
        mdl_hw = _get_modified_hw_multiple_of(hw=dataset_hw, multiple_of=multiple_of)
        dataset_cfg.target_size = mdl_hw

        #　入力データに応じて、次元を設定
        if dataset_cfg.ev_representation == 'event_frame':
            input_dim = 3
        else:
            input_dim = 20 
        mdl_cfg.backbone.input_dim = input_dim

        ##データセットのクラス数
        class_len_map = {
            'gen1': 2,
            'gen4': 3,
            'dsec': 8
        }
        mdl_cfg.head.num_classes = class_len_map[dataset_name]
        print('num_class', mdl_cfg.head.num_classes)


        if 'rvt' in mdl_name:  # 'rvt' が含まれているかどうかで判定
            backbone_cfg = mdl_cfg.backbone
            backbone_name = backbone_cfg.name
            if backbone_name in {'RVT'}:
            
                attention_cfg = backbone_cfg.stage.attention
                partition_size = tuple(x // (32 * partition_split_32) for x in mdl_hw)
                assert (mdl_hw[0] // 32) % partition_size[0] == 0, f'{mdl_hw[0]=}, {partition_size[0]=}'
                assert (mdl_hw[1] // 32) % partition_size[1] == 0, f'{mdl_hw[1]=}, {partition_size[1]=}'
                print(f'Set partition sizes: {partition_size}')
                attention_cfg.partition_size = partition_size
            else:
                print(f'{backbone_name=} not available')
                raise NotImplementedError
        



def _get_modified_hw_multiple_of(hw: Tuple[int, int], multiple_of: int) -> Tuple[int, ...]:
    assert len(hw) == 2
    assert isinstance(multiple_of, int)
    assert multiple_of >= 1
    if multiple_of == 1:
        return hw
    new_hw = tuple(math.ceil(x / multiple_of) * multiple_of for x in hw)
    return new_hw
