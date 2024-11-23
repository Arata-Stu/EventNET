import os
from typing import Tuple

import math
from omegaconf import DictConfig, open_dict


def dynamically_modify_train_config(config: DictConfig):
    with open_dict(config):
        
        dataset_cfg = config.dataset
        dataset_name = dataset_cfg.name
        assert dataset_name in {'gen1', 'gen4', 'dsec'}

        dataset_hw = dataset_cfg.orig_size

        mdl_cfg = config.model
        mdl_name = mdl_cfg.name

        #32の倍数になるようにheight widthを調整
        partition_split_32 = mdl_cfg.backbone.partition_split_32
        assert partition_split_32 in (1, 2, 4)
        multiple_of = 32 * partition_split_32
        mdl_hw = _get_modified_hw_multiple_of(hw=dataset_hw, multiple_of=multiple_of)
        dataset_cfg.target_size = mdl_hw
        mdl_cfg.backbone.in_res_hw = mdl_hw

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
        elif 'sast' in mdl_name:  # 'rvt' が含まれているかどうかで判定
            backbone_cfg = mdl_cfg.backbone
            backbone_name = backbone_cfg.name
            if backbone_name in {'SAST'}:
            
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


## loop内でconfigを変更する場合
def modify_config(config, **kwargs):
    """
    引数として渡されたパラメータに応じて、configの値を動的に変更する関数。
    ネストされたキーはドット区切りで指定する。
    
    Args:
        config (DictConfig): 元の設定を含むconfigオブジェクト
        **kwargs: 変更するパラメータ。ネストされたキーはドットで区切る
    
    Example:
        modify_config(config, dataset__delta_t_ms=20, dataset__augmentation__prob_hflip=0.3)
    """
    for key, value in kwargs.items():
        keys = key.split('__')  # ネストされたキーはドットで区切って指定
        target = config
        for subkey in keys[:-1]:  # 最後のキー以外を順にたどる
            target = target.get(subkey)
            if target is None:
                raise KeyError(f"Configに指定されたキー '{subkey}' が存在しません")
        
        # 最後のキーに対応する値を書き換え
        target[keys[-1]] = value
    return config
