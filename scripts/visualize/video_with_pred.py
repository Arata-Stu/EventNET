import argparse
from utils.visualize import save_sequence_with_pred
from omegaconf import OmegaConf
from config.modifier import dynamically_modify_train_config
from modules.fetch import fetch_model_module, fetch_data_module

import sys
import os

# プロジェクトのルートディレクトリを指定
project_root = os.path.abspath(os.path.join(os.getcwd(), './../..'))
sys.path.append(project_root)

# 引数を設定
parser = argparse.ArgumentParser(description="推論結果を動画として保存")
parser.add_argument('--model', type=str, required=True, help='Path to model configuration file')
parser.add_argument('--exp', type=str, required=True, help='Path to experiment configuration file')
parser.add_argument('--dataset', type=str, required=True, help='Path to dataset configuration file')
parser.add_argument('--output_file', type=str, default='output.mp4', help='保存する動画ファイルの名前')
args = parser.parse_args()

# Configを読み込み、マージ
config_paths = [args.model, args.dataset, args.exp]
configs = [OmegaConf.load(path) for path in config_paths]
merged_conf = OmegaConf.merge(*configs)
dynamically_modify_train_config(config=merged_conf)

t_ms = merged_conf.dataset.tau_ms

# データとモデルを設定
data = fetch_data_module(config=merged_conf)
model = fetch_model_module(config=merged_conf)
data.setup('fit')
model.setup('fit')
pred_model = model.model
pred_model.eval()

# 動画の保存処理
model_type = merged_conf.model.type
save_sequence_with_pred(
    dataloader=data.val_dataloader(),
    t_ms=t_ms,
    output_file=args.output_file,
    model=pred_model,
    conf_thre=0.1,
    nms_thre=0.45,
    model_type=model_type
)
