import cv2
import numpy as np
import torch
from typing import Any
from functools import partial
from torch.utils.data import DataLoader
from .yolox_utils import to_yolox, postprocess
from modules.utils.rnn_state import RNNStates

def save_sequence_as_video_from_dataloader(dataloader: DataLoader, t_ms: int, output_file: str, mode='train'):
    fps = 1000 / t_ms

    # 最初のバッチからサイズを取得
    dataloader_iter = iter(dataloader)
    first_batch = next(dataloader_iter)
    B, L, ch, h, w = first_batch['events'].shape
    size = (w, h)

    # VideoWriterのセットアップ
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, size, isColor=True)

    # 各バッチを逐次取得
    for batch_idx, batch in enumerate(dataloader_iter):
        events = batch['events']
        labels = batch['labels']
        mask = batch.get('mask', None)

        # YOLOX形式のバウンディングボックスを取得
        outputs = to_yolox(labels, mode=mode)

        # バッチ内の処理
        for b in range(events.shape[0]):
            for t in range(events.shape[1]):
                if mask is not None and mask[b, t].item() == 0:
                    continue

                # フレーム変換と描画
                frame = events[b, t]
                img_uint = np.transpose(frame.numpy(), (1, 2, 0)).astype('uint8').copy()

                # バウンディングボックスの描画
                for bbox in outputs[b, t]:
                    if torch.all(bbox == 0):  # 無効なバウンディングボックスはスキップ
                        continue

                    # `mode` に応じたバウンディングボックスの座標取得
                    if mode == 'train':
                        cls, cx, cy, w, h = bbox
                        x1 = int(cx - w / 2)
                        y1 = int(cy - h / 2)
                        x2 = int(cx + w / 2)
                        y2 = int(cy + h / 2)
                    elif mode in ['val', 'test']:
                        x1, y1, w, h, cls = bbox
                        x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)

                    # バウンディングボックスとクラスラベルをフレームに描画
                    cv2.rectangle(img_uint, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(img_uint, f"Cls: {int(cls)}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # フレーム書き込み
                video_writer.write(img_uint)

    video_writer.release()
    print(f"動画が保存されました: {output_file}")


def save_sequence_with_pred(dataloader: DataLoader, t_ms: int, output_file: str, model: Any, type: str = 'gen1', mode='train'):
    fps = 1000 / t_ms

    assert type in ['gen1', 'gen4', 'dsec'], f"Invalid model type: {type}"

    classes_map = {
        'gen1': 2,
        'gen4': 3,
        'dsec': 8
    }

    # VideoWriterのセットアップ（バッチごとではなく全データに対して1つのファイル）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None

    for batch in dataloader:
        sequence_events = batch['events']  # [batch, sequence_len, ch, h, w]
        sequence_labels = batch['labels']  # [batch, sequence_len, num, bbox]
        sequence_timestamps = batch['timestamps']
        is_first_sample = batch['is_start_sequence']  # list[] * batch
        sequence_is_padded_mask = batch['mask']  # [batch, sequence_len]

        # YOLOX形式のバウンディングボックスを取得
        sequence_targets = to_yolox(sequence_labels, mode=mode)

        B, L, ch, h, w = sequence_events.shape
        size = (w, h)

        if video_writer is None:
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, size, isColor=True)

        post_process = partial(postprocess,
                               num_classes=classes_map[type],
                               conf_thre=0.1,
                               nms_thre=0.45,
                               class_agnostic=False)

        worker_id = 0
        rnn_state = RNNStates()
        rnn_state.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        prev_states = rnn_state.get_states(worker_id=worker_id)

        for t in range(L):
            # 各時系列ステップのデータ取得
            events = sequence_events[:, t, :, :, :]
            targets = sequence_targets[:, t, :, :] if t < sequence_targets.shape[1] else None  # t の範囲内確認
            token_mask = sequence_is_padded_mask[:, t]
            time = sequence_timestamps[:, t]
            
            if targets is None or targets.shape[1] == 0:
                print("Skipping frame due to empty or out-of-bounds targets.")
                continue

            # バッチ内の処理
            for b in range(events.shape[0]):
                if token_mask is not None and token_mask.item() == 0:
                    continue

                frame = events.squeeze(0)
                img_uint = np.transpose(frame.numpy(), (1, 2, 0)).astype('uint8').copy()

                # 推論を実行
                events = events.float()
                preds, states = model(events, prev_states)  # フレームをモデルに入力
                prev_states = states
                processed_preds = post_process(prediction=preds)

                # 予測バウンディングボックスの描画
                if processed_preds[0] is not None:
                    pred_bboxes = processed_preds[0].cpu().numpy()
                    for pred_box in pred_bboxes:
                        try:
                            x1, y1, x2, y2, conf, conf_class, cls = pred_box
                            start_point = (int(x1), int(y1))
                            end_point = (int(x2), int(y2))

                            # 青色で太さ2の矩形描画 (予測)
                            cv2.rectangle(img_uint, start_point, end_point, (255, 0, 0), 2)
                            cv2.putText(img_uint, f"Pred: {int(cls)}, Conf: {conf:.2f}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        except Exception as e:
                            print(f"Error drawing prediction rectangle at batch {b}, box {pred_box}: {e}")

                # 元のラベルのバウンディングボックス描画
                if t < targets.shape[1] and b < targets.shape[0]:  # t と b の範囲を確認
                    for bbox in targets[b, t]:
                        if torch.all(bbox == 0):  # 無効なバウンディングボックスはスキップ
                            continue

                        # `mode` に応じたバウンディングボックスの座標取得
                        if mode == 'train':
                            cls, cx, cy, w, h = bbox
                            x1 = int(cx - w / 2)
                            y1 = int(cy - h / 2)
                            x2 = int(cx + w / 2)
                            y2 = int(cy + h / 2)
                        elif mode in ['val', 'test']:
                            x1, y1, w, h, cls = bbox
                            x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)

                        # 黄色で太さ2の矩形描画 (元のラベル)
                        cv2.rectangle(img_uint, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(img_uint, f"Cls: {int(cls)}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # フレーム書き込み
                video_writer.write(img_uint)

    video_writer.release()
    print(f"動画が保存されました: {output_file}")
