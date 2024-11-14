import sys
import os
# プロジェクトのルートディレクトリを指定
project_root = os.path.abspath(os.path.join(os.getcwd(), './../..'))
sys.path.append(project_root)

import argparse
from utils.visualize import save_sequence_with_pred
from omegaconf import OmegaConf
from config.modifier import dynamically_modify_train_config
from modules.fetch import fetch_model_module, fetch_data_module


def get_config_paths_from_ckpt(ckpt_path):
    # ckpt_pathからベースディレクトリを取得
    train_dir = os.path.dirname(ckpt_path)
    
    # merged_config.yaml のパスを自動で取得
    config_path = os.path.join(train_dir, "merged_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path}が見つかりません")
    
    return config_path, train_dir

if __name__ == "__main__":
    # 引数を設定
    parser = argparse.ArgumentParser(description="推論結果を動画として保存")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output file")
    args = parser.parse_args()
    
    # ckpt_pathとtrain_dirを取得
    config_path, train_dir = get_config_paths_from_ckpt(args.ckpt_path)
    
    # merged_config.yaml のパスを自動で取得
    config_path = os.path.join(train_dir, "merged_config.yaml")
    merged_conf = OmegaConf.load(config_path)
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
        type='gen1',
        mode='val'
    )
