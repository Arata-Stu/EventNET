from test import main as test  # test.py の main 関数をインポート
import yaml
import argparse

# 引数を解析
parser = argparse.ArgumentParser(description="Testing loop with multiple checkpoint paths")
parser.add_argument("--config", type=str, required=True, help="Path to the ckpt_paths.yaml file containing checkpoint paths list")
args = parser.parse_args()

# 指定された YAML ファイルを読み込む
with open(args.config, 'r') as file:
    config_list = yaml.safe_load(file)

# ckpt_paths のリストを取得
ckpt_paths = config_list['ckpt_paths']

# 各 ckpt_path をループしてテスト実行
for ckpt_path in ckpt_paths:
    print(f"Testing with checkpoint: {ckpt_path}")
    test(ckpt_path)
