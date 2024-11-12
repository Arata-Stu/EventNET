from itertools import product
import argparse
import yaml
import os
from .train import main as train

# 引数を解析
parser = argparse.ArgumentParser(description="Training loop with multiple configuration combinations")
parser.add_argument("--config", type=str, required=True, help="Path to the list.yaml file containing config file lists")
args = parser.parse_args()

# 指定された YAML ファイルを読み込む
with open(args.config, 'r') as file:
    config_list = yaml.safe_load(file)

# 各設定ファイルのパスを取得
model_configs = config_list['model_configs']
exp_configs = config_list['exp_configs']
dataset_configs = config_list['dataset_configs']

# すべての組み合わせをループ
for model_config, exp_config, dataset_config in product(model_configs, exp_configs, dataset_configs):
    print(f"Training with model config: {model_config}, experiment config: {exp_config}, dataset config: {dataset_config}")
    # 各組み合わせで train.py の main 関数を呼び出し、新規トレーニングを実行
    model_config_path = os.path.join('../../', model_config)
    exp_config_path = os.path.join('../../', exp_config)
    dataset_config_path = os.path.join('../../', dataset_config)
    train(model_config, exp_config, dataset_config)
