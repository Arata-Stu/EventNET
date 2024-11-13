import argparse
import yaml
import os
from omegaconf import OmegaConf
from test import main as test


# 引数を解析
parser = argparse.ArgumentParser(description="Testing loop with multiple checkpoint paths")
parser.add_argument("--config", type=str, required=True, help="Path to the ckpt_paths.yaml file containing checkpoint paths list")
args = parser.parse_args()

# 指定された YAML ファイルを読み込む
config_list_path = args.config
config_list = OmegaConf.load(config_list_path)

# ckpt_paths のリストを取得
ckpt_paths = config_list['ckpt_paths']

# 各 ckpt_path をループしてテスト実行
for ckpt_path in ckpt_paths:
    test(ckpt_path)
