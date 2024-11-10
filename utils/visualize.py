import cv2
import numpy as np
import torch
from typing import Any
from torch.utils.data import DataLoader
from .yolox_utils import to_yolox

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
