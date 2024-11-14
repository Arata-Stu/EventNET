import sys
sys.path.append('./../../')

import argparse
from omegaconf import OmegaConf
from config.modifier import modify_config
from train import main as train

def flatten_dict(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="Training loop with multiple configuration combinations")
parser.add_argument("--config", type=str, required=True, help="Path to the config_list.yaml file containing config paths and parameters")
args = parser.parse_args()

# config_list.yamlのパスを読み込み
config_list_path = args.config
config_list = OmegaConf.load(config_list_path)

# 各設定ファイルのパスを取得し、configを展開
model_config = OmegaConf.load(config_list['model_configs'][0])
exp_config = OmegaConf.load(config_list['exp_configs'][0])
dataset_config_base = OmegaConf.load(config_list['dataset_configs'][0])

# パラメータの組み合わせを取得
parameters = config_list['parameters']

# 各パラメータセット（param1, param2, ...）でループ
for param_name, param in parameters.items():
    # dataset_config_baseをコピーし、必要なパラメータを更新
    dataset_config = dataset_config_base.copy()  # 元の設定を保持し、変更に使うコピーを作成
    
    # `param` をフラット化し、動的に `modify_config` に渡して更新
    flat_param = flatten_dict(param)  # ネストされた構造をフラット化
    modified_config = modify_config(dataset_config, **{f"dataset__{key}": value for key, value in flat_param.items()})

    # 統合したconfigを作成
    merged_conf = OmegaConf.merge(model_config, exp_config, modified_config)

    # 訓練実行
    print(f"Training with parameters: {param_name}")
    train(merged_conf)  # `train` に `merged_conf` を直接渡す
