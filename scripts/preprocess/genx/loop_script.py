import os
import yaml
import time
import argparse
import copy

# 既存の前処理スクリプトのmain関数を呼び出す
from preprocess import main  # main関数が記載されたスクリプトの名前に変更

def process_with_parameters(base_config, parameters):
    for param in parameters:
        config = copy.deepcopy(base_config)  # 基本configをコピーして変更
        config['delta_t_ms'] = param['delta_t_ms']
        config['tau_ms'] = param['tau_ms']
        print(f"Processing with delta_t_ms={param['delta_t_ms']} and tau_ms={param['tau_ms']}")

        start_time = time.time()
        
        # 前処理のメイン関数を実行
        main(config)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Completed processing with delta_t_ms={param['delta_t_ms']} and tau_ms={param['tau_ms']} in {elapsed_time:.2f} seconds\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process with multiple delta and tau values from list.yaml")
    parser.add_argument('--list', type=str, required=True, help="Path to the list.yaml file containing base config file and parameters")
    args = parser.parse_args()

    # list.yamlファイルを読み込み、base_configとparametersリストを取得
    with open(args.list, 'r') as f:
        config_data = yaml.safe_load(f)  # base configファイルのパスとパラメータリストを読み込み
        base_config_path = config_data['base_config']
        parameters = config_data['parameters']

    # base_configを読み込み
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # 各deltaとtauの組み合わせで処理
    process_with_parameters(base_config, parameters)
