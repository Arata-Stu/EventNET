import yaml
from omegaconf import OmegaConf
from config.modifier import dynamically_modify_train_config
from modules.fetch import fetch_data_module, fetch_model_module

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
import os
import datetime
import argparse

def main(model_config, exp_config, dataset_config):
    base_save_dir = './result'
    
    # 実行時のタイムスタンプを付与して、一意のディレクトリ名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 各 YAML ファイルを読み込んで OmegaConf にマージ
    configs = [OmegaConf.load(path) for path in [model_config, exp_config, dataset_config]]
    merged_conf = OmegaConf.merge(*configs)
    dynamically_modify_train_config(merged_conf)
    
    # dataset.name, model.name, ev_representation, ev_delta_t を取得
    dataset_name = merged_conf.dataset.name
    model_name = merged_conf.model.name
    ev_representation = merged_conf.dataset.ev_representation
    delta_t_ms = merged_conf.dataset.delta_t_ms
    
    # ev_representation と ev_delta_t を組み合わせたディレクトリ名
    event_rep_dir = f"{ev_representation}-dt{delta_t_ms}"
    
    # ディレクトリの階層構造を作成
    save_dir = os.path.join(base_save_dir, dataset_name, model_name, event_rep_dir, timestamp, 'train')
    os.makedirs(save_dir, exist_ok=True)  # 保存ディレクトリを作成
    
    # 統合されたconfigを保存
    merged_config_path = os.path.join(save_dir, 'merged_config.yaml')
    with open(merged_config_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(merged_conf, resolve=True), f)
    
    # データモジュールとモデルモジュールのインスタンスを作成
    data = fetch_data_module(merged_conf)
    data.setup('fit')
    model = fetch_model_module(merged_conf)
    model.setup('fit')
    
    # コールバックの設定
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,  # save_dirにチェックポイントを保存
            filename='{epoch:02d}-{val_AP:.2f}',
            monitor='val_AP',  # 基準とする量
            mode="max", 
            save_top_k=3,  # 保存するトップkのチェックポイント
            save_last=True, 
        ),
        # 学習率のモニターを追加
        LearningRateMonitor(logging_interval='step')
    ]

    # TensorBoard Loggerもsave_dirに対応させる
    logger = pl_loggers.TensorBoardLogger(
        save_dir=save_dir,  # ckptの保存ディレクトリに合わせる
        name='',  # nameを空にすることで、サブディレクトリを作成しない
        version='',
    )

    

    train_cfg = merged_conf.experiment.training
    # トレーナーを設定
    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        max_steps=train_cfg.max_steps,
        logger=logger,  # Loggerに対応させる
        callbacks=callbacks,
        accelerator='gpu',
        precision=train_cfg.precision, 
        devices=[0],  # 使用するGPUのIDのリスト
        benchmark=True,  # cudnn.benchmarkを使用して高速化
        profiler= 'simple',
    )

    # モデルの学習を実行
    trainer.fit(model, datamodule=data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with specified YAML config files')
    parser.add_argument('--model', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--exp', type=str, required=True, help='Path to experiment configuration file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset configuration file')

    args = parser.parse_args()
    main(args.model, args.exp, args.dataset)
