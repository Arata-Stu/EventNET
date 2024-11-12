import os

def modify_paths(config):
    # dataset.data_dir に ../ を追加
    config['dataset']['data_dir'] = os.path.join('..', config['dataset']['data_dir'])

    # dataset.split_config が存在する場合に ../ を追加
    if 'split_config' in config['dataset']:
        config['dataset']['split_config'] = os.path.join('..', config['dataset']['split_config'])

    return config