import yaml
from omegaconf import OmegaConf
from config.modifier import dynamically_modify_train_config
from modules.fetch import fetch_data_module, fetch_model_module

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
import argparse

def get_config_paths_from_ckpt(ckpt_path):
    # ckpt_pathからベースディレクトリを取得
    train_dir = os.path.dirname(ckpt_path)
    
    # merged_config.yaml のパスを自動で取得
    config_path = os.path.join(train_dir, "merged_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path}が見つかりません")
    
    return config_path, train_dir

def main(ckpt_path):
    # 設定ファイルとベースディレクトリを取得
    config_path, train_dir = get_config_paths_from_ckpt(ckpt_path)

    save_dir = os.path.join(train_dir, 'test')
    os.makedirs(save_dir, exist_ok=True)  # 保存ディレクトリを作成

    # YAML ファイルを読み込んで OmegaConf に変換
    merged_conf = OmegaConf.load(config_path)
    dynamically_modify_train_config(merged_conf)

    # データモジュールとモデルモジュールのインスタンスを作成
    data = fetch_data_module(merged_conf)
    data.setup('test')
    model = fetch_model_module(merged_conf)
    model.setup('test')

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])

    # TensorBoard Loggerもsave_dirに対応させる
    logger = pl_loggers.TensorBoardLogger(
        save_dir=save_dir,  # ckptの保存ディレクトリに合わせる
        name='',  # nameを空にすることで、サブディレクトリを作成しない
        version='',
    )

    # トレーナーを設定
    trainer = pl.Trainer(
        logger=logger,  # Loggerに対応させる
        callbacks=None,
        accelerator='gpu',
        devices=[0],  # 使用するGPUのIDのリスト
        benchmark=True,  # cudnn.benchmarkを使用して高速化
    )

    # モデルのテストを実行
    trainer.test(model, datamodule=data)

if __name__ == '__main__':
    # argparseでコマンドライン引数を取得
    parser = argparse.ArgumentParser(description="Model testing script")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the checkpoint file")
    
    # 引数をパース
    args = parser.parse_args()

    # ckpt_pathをmainに渡す
    main(args.ckpt_path)
