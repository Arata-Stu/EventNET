import yaml
from omegaconf import OmegaConf
from config.modifier import dynamically_modify_train_config
from modules.fetch import fetch_data_module, fetch_model_module

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
import os
import argparse
from datetime import datetime

def create_resume_config_from_ckpt(ckpt_path, max_epochs):
    # ckpt_path から merged_config.yaml のパスを推測
    config_dir = os.path.dirname(ckpt_path)
    merged_config_path = os.path.join(config_dir, 'merged_config.yaml')
    
    if not os.path.exists(merged_config_path):
        raise FileNotFoundError(f"Config file not found at expected path: {merged_config_path}")

    # 元の merged_config.yaml を読み込む
    merged_conf = OmegaConf.load(merged_config_path)
    
    # max_epochs を上書き
    merged_conf.experiment.training.max_epochs = max_epochs

    # 再開用の新しいディレクトリを生成
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_config_dir = os.path.join(config_dir, f"resume_{timestamp}")
    os.makedirs(new_config_dir, exist_ok=True)

    # 新しい設定ファイル名を作成
    new_config_path = os.path.join(new_config_dir, 'merged_config.yaml')
    
    # 上書きした設定を新しいファイルに保存
    with open(new_config_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(merged_conf, resolve=True), f)

    print(f"New resume configuration saved at: {new_config_path}")
    return new_config_path, merged_conf

def main(resume_ckpt, max_epochs):
    # ckpt_path から新しい max_epochs 設定を含む設定ファイルを生成
    new_config_path, merged_conf = create_resume_config_from_ckpt(resume_ckpt, max_epochs)
    dynamically_modify_train_config(merged_conf)
    
    # データセットやモデル情報を取得
    data = fetch_data_module(merged_conf)
    data.setup('fit')
    model = fetch_model_module(merged_conf)
    model.setup('fit')
    
    # 新しいログ保存用のディレクトリ
    save_dir = os.path.dirname(new_config_path)
    
    # コールバックの設定
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,  # 新しいディレクトリにチェックポイントを保存
            filename='{epoch:02d}-{val_AP:.2f}',
            monitor='val_AP',
            mode="max", 
            save_top_k=3,
            save_last=True, 
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # 新しいディレクトリにTensorBoard Loggerを設定
    logger = pl_loggers.TensorBoardLogger(
        save_dir=save_dir,  # 新しい再開用ディレクトリに保存
        name='',  # サブディレクトリを作成しない
        version=''
    )

    train_cfg = merged_conf.experiment.training
    # トレーナーの設定
    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        max_steps=train_cfg.max_steps,
        logger=logger,
        callbacks=callbacks,
        accelerator='gpu',
        precision=train_cfg.precision, 
        devices=[0],  # 使用するGPUのIDのリスト
        benchmark=True,  # cudnn.benchmark を使用して高速化
    )

    # 再開トレーニングの実行
    trainer.fit(model, datamodule=data, ckpt_path=resume_ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resume training from a specified checkpoint')
    parser.add_argument('--resume_ckpt', type=str, required=True, help='Path to checkpoint to resume from')
    parser.add_argument('--max_epochs', type=int, required=True, help='New max epochs for resumed training')

    args = parser.parse_args()
    main(args.resume_ckpt, args.max_epochs)
