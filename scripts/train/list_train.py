import os
import argparse
import yaml
from omegaconf import OmegaConf, DictConfig
from train import main as train

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="Training loop with multiple configuration combinations")
parser.add_argument("--config", type=str, required=True, help="Path to the config_list.yaml file containing config paths and parameters")
args = parser.parse_args()

# 引数で指定されたconfig_list.yamlのパスを読み込み
config_list_path = args.config
config_list = OmegaConf.load(config_list_path)

# 各設定ファイルのパスを取得
model_config_path = config_list['model_configs'][0]
exp_config_path = config_list['exp_configs'][0]
dataset_config_path = config_list['dataset_configs'][0]

# パラメータの組み合わせを取得
parameters = config_list['parameters']

# 各パラメータの組み合わせでループ
for param in parameters:
    delta_t_ms = param['delta_t_ms']
    tau_ms = param['tau_ms']
    
    # dataset_configの読み込みと設定の更新
    dataset_config = OmegaConf.load(dataset_config_path)
    dataset_config.dataset.delta_t_ms = delta_t_ms
    dataset_config.dataset.tau_ms = tau_ms
    
    # 一時的な設定ファイルとして保存
    temp_dataset_config_path = f"temp_dataset_config_delta{delta_t_ms}_tau{tau_ms}.yaml"
    OmegaConf.save(dataset_config, temp_dataset_config_path)

    # 訓練実行
    print(f"Training with delta_t_ms: {delta_t_ms}, tau_ms: {tau_ms}")
    train(model_config=model_config_path, 
          exp_config=exp_config_path, 
          dataset_config=temp_dataset_config_path)

    # 一時ファイルの削除
    os.remove(temp_dataset_config_path)
