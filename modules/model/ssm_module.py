import torch
import torch.nn as nn
import pytorch_lightning as pl
from functools import partial
from omegaconf import DictConfig

from models.model import build_model
from modules.utils.rnn_state import RNNStates
from utils.eval.prophesee.evaluator import PropheseeEvaluator
from utils.eval.prophesee.io.box_loading import to_prophesee
from utils.yolox_utils import postprocess, to_yolox


class SMMModule(pl.LightningModule):

    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config

        if self.full_config.dataset.name == "gen1":
            from data.dataset.genx.classes import  GEN1_CLASSES as CLASSES
            self.height, self.width = 240, 304
        elif self.full_config.dataset.name == "gen4":
            from data.dataset.genx.classes import GEN4_CLASSES as CLASSES
            self.height, self.width = 360, 640
        elif self.full_config.dataset.name == "dsec":
            from data.dataset.dsec.detection.classes import CLASSES as CLASSES
            self.height, self.width = 480, 640
        
        self.classes = CLASSES  # クラスを保持
        self.model = build_model(model_config=full_config.model)

        self.post_process = partial(postprocess,
                                    num_classes=full_config.model.head.num_classes,
                                    conf_thre=full_config.model.postprocess.conf_thre,
                                    nms_thre=full_config.model.postprocess.nms_thre,
                                    class_agnostic=False)

       
    def setup(self, stage):
        self.started_training = True
        
        if stage == 'fit':
            self.started_training = False
            self.val_evaluator = PropheseeEvaluator(dataset=self.full_config.dataset.name, 
                                                downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            
            self.train_rnn_state = RNNStates()
            self.val_rnn_state = RNNStates()
        elif stage == 'test':
            self.test_evaluator = PropheseeEvaluator(dataset=self.full_config.dataset.name, 
                                                downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            
            self.test_rnn_state = RNNStates()
        
        
    def forward(self, x, targets=None):
        return self.model(x, targets)
    
    def training_step(self, batch, batch_idx):
        self.started_training = True
        self.model.train()

        # バッチデータ取得
        sequence_events = batch['events']  # [batch, sequence_len, ch, h, w]
        sequence_labels = batch['labels']  # [batch, sequence_len, num, bbox]
        sequence_timestamps = batch['timestamps']
        is_first_sample = batch['is_start_sequence']  # list[] * batch
        sequence_is_padded_mask = batch['mask']  # [batch, sequence_len]

        sequence_events = sequence_events.permute(1, 0, 2, 3, 4)  # -> [L, B, C, H, W]
        

        sequence_targets = to_yolox(sequence_labels, mode='train')
        
        # RNN状態の初期化
        worker_id = 0
        self.train_rnn_state.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        prev_states = self.train_rnn_state.get_states(worker_id=worker_id)

        batch_size, sequence_length, ch, height, width = sequence_events.shape

        all_saved_data = []
        padding = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=sequence_targets.device)

        backbone_features, states = self.model.forward_backbone(
        x=sequence_events.to(dtype=self.dtype),
        previous_states=prev_states,  # SMMでは初期状態不要
        token_mask=None
        )

        self.train_rnn_state.save_states_and_detach(worker_id=worker_id, states=prev_states)

        padding = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=sequence_targets.device)
        valid_batch_indices = [
            i for i, batch in enumerate(sequence_targets.view(-1, sequence_targets.size(-1)))
            if (batch != padding).any()
        ]

        if valid_batch_indices:
            # 有効なデータを抽出
            valid_targets = sequence_targets.view(-1, sequence_targets.size(-2), sequence_targets.size(-1))[valid_batch_indices]
            valid_backbone_features = {
                key: feature.view(-1, feature.size(-1))[valid_batch_indices]
                for key, feature in backbone_features.items()
            }

            # 損失計算
            loss = self.model.forward_detect(
                backbone_features=valid_backbone_features,
                targets=valid_targets
            )

            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        else:
            return None


    def on_train_epoch_end(self):
        #訓練を再開した時の、ckptコールバックのエラーを回避するため
        # if 'val_AP' not in self.trainer.callback_metrics:
        #     self.log('val_AP', 0.0, on_epoch=True, prog_bar=True, logger=True)
        pass

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        # データ取得
        sequence_events = batch["events"]  # [B, L, C, H, W]
        sequence_labels = batch["labels"]  # [B, L, num, bbox]
        is_first_sample = batch['is_start_sequence']  # list[] * batch
        sequence_timestamps = batch["timestamps"]
        sequence_is_padded_mask = batch["mask"]  # [B, L]

        if sequence_labels.shape[2] == 0:
            return  # ラベルが空の場合はスキップ
        
        worker_id = 0
        self.val_rnn_state.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        prev_states = self.val_rnn_state.get_states(worker_id=worker_id)

        # データを(L, B, C, H, W)形式に変換
        sequence_events = sequence_events.permute(1, 0, 2, 3, 4)  # [L, B, C, H, W]
        sequence_targets = to_yolox(sequence_labels, mode="val")

        # モデルに時系列データを一括入力
        backbone_features, states = self.model.forward_backbone(
            x=sequence_events.to(dtype=self.dtype),
            previous_states=prev_states,  # RNN状態は不要
            token_mask=None
        )

        prev_states = states

        # ラベルの存在確認と有効データの抽出
        padding = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=sequence_targets.device)
        valid_indices = [
            (l, b) for l in range(sequence_targets.size(0))
            for b in range(sequence_targets.size(1))
            if (sequence_targets[l, b, :, :] != padding).any()
        ]

        if valid_indices:
            # 有効データをバッチ化
            valid_targets = torch.stack(
                [sequence_targets[l, b, :, :] for l, b in valid_indices]
            )
            valid_backbone_features = {
                key: torch.stack(
                    [backbone_features[key][l, b, :] for l, b in valid_indices]
                )
                for key in backbone_features.keys()
            }

            # 推論結果の取得
            preds = self.model.forward_detect(
                backbone_features=valid_backbone_features
            )
            processed_preds = self.post_process(prediction=preds)

            # Prophesee形式に変換
            loaded_labels_proph, yolox_preds_proph = to_prophesee(
                loaded_label_tensor=valid_targets,
                label_timestamps=sequence_timestamps.view(-1),  # フラット化
                yolox_pred_list=processed_preds,
            )

            # 評価器にラベルと予測を追加
            if hasattr(self, "val_evaluator"):
                self.val_evaluator.add_labels(loaded_labels_proph)
                self.val_evaluator.add_predictions(yolox_preds_proph)

            return 

    def on_validation_epoch_end(self):
        if self.started_training:
            if self.val_evaluator.has_data():
                metrics = self.val_evaluator.evaluate_buffer(img_height=self.height,
                                                            img_width=self.width)
                for k, v in metrics.items():
                    # val_APのみプログレスバーに表示する
                    prog_bar_flag = True if k == "AP" else False
                    self.log(f'val_{k}', v, on_epoch=True, prog_bar=prog_bar_flag, logger=True)
                self.val_evaluator.reset_buffer()

    def test_step(self, batch, batch_idx):
        self.model.eval()

        # データ取得
        sequence_events = batch["events"]  # [B, L, C, H, W]
        sequence_labels = batch["labels"]  # [B, L, num, bbox]
        is_first_sample = batch['is_start_sequence']  # list[] * batch
        sequence_timestamps = batch["timestamps"]
        sequence_is_padded_mask = batch["mask"]  # [B, L]

        if sequence_labels.shape[2] == 0:
            return  # ラベルが空の場合はスキップ
        
        worker_id = 0
        self.test_rnn_state.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        prev_states = self.test_rnn_state.get_states(worker_id=worker_id)

        # データを(L, B, C, H, W)形式に変換
        sequence_events = sequence_events.permute(1, 0, 2, 3, 4)  # [L, B, C, H, W]
        sequence_targets = to_yolox(sequence_labels, mode="test")

        # モデルに時系列データを一括入力
        backbone_features, states = self.model.forward_backbone(
            x=sequence_events.to(dtype=self.dtype),
            previous_states=prev_states,  # RNN状態は不要
            token_mask=None
        )
        prev_states = states

        # ラベルの存在確認と有効データの抽出
        padding = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=sequence_targets.device)
        valid_indices = [
            (l, b) for l in range(sequence_targets.size(0))
            for b in range(sequence_targets.size(1))
            if (sequence_targets[l, b, :, :] != padding).any()
        ]

        if valid_indices:
            # 有効データをバッチ化
            valid_targets = torch.stack(
                [sequence_targets[l, b, :, :] for l, b in valid_indices]
            )
            valid_backbone_features = {
                key: torch.stack(
                    [backbone_features[key][l, b, :] for l, b in valid_indices]
                )
                for key in backbone_features.keys()
            }

            # 推論結果の取得
            preds = self.model.forward_detect(
                backbone_features=valid_backbone_features
            )
            processed_preds = self.post_process(prediction=preds)

            # Prophesee形式に変換
            loaded_labels_proph, yolox_preds_proph = to_prophesee(
                loaded_label_tensor=valid_targets,
                label_timestamps=sequence_timestamps.view(-1),  # フラット化
                yolox_pred_list=processed_preds,
            )

            # 評価器にラベルと予測を追加
            if hasattr(self, "val_evaluator"):
                self.test_evaluator.add_labels(loaded_labels_proph)
                self.test_evaluator.add_predictions(yolox_preds_proph)

            return 

    def on_test_epoch_end(self):
        if self.started_training:
            if self.test_evaluator.has_data():
                metrics = self.test_evaluator.evaluate_buffer(img_height=self.height,
                                                            img_width=self.width)
                for k, v in metrics.items():
                    # val_APのみプログレスバーに表示する
                    prog_bar_flag = True if k == "AP" else False
                    self.log(f'test_{k}', v, on_epoch=True, prog_bar=prog_bar_flag, logger=True)
                self.test_evaluator.reset_buffer()
        
        
    def configure_optimizers(self):
        lr = self.full_config.experiment.training.learning_rate
        weight_decay = self.full_config.experiment.training.weight_decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.full_config.experiment.training.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}


