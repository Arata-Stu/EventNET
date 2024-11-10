import cv2
import numpy as np
from typing import Any
from torch.utils.data import DataLoader
import time

def save_sequence_as_video_from_dataset(dataset: Any, start_index: int, end_index: int, t_ms: int, output_file: str):
    """
    指定範囲のサンプルを動画として保存する関数。
    
    Args:
        dataset: チェックするデータセット
        start_index (int): 開始インデックス
        end_index (int): 終了インデックス
        t_ms (int): 各フレームの表示時間 (ミリ秒)
        output_file (str): 出力動画ファイル名
    """
    # FPS計算
    fps = 1000 / t_ms
    
    # サンプルデータから画像サイズを取得
    sample = dataset[start_index]
    _, h, w = sample['events'][0].shape  # (ch, h, w)のshapeを取得
    size = (w, h)
    
    # VideoWriterのセットアップ
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4フォーマット
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, size, isColor=True)

    for index in range(start_index, end_index):
        sample = dataset[index]
        events = sample['events']
        labels = sample['labels']
        mask = sample.get('mask', None)  # マスクデータを取得、なければNone

        # 各時刻のイベントデータを順番にプロットし、フレームとして追加
        for t, (frame, bbox) in enumerate(zip(events, labels)):
            # maskが0の場合は無視するためスキップ
            if mask is not None and mask[t].item() == 0:
                continue  # 無効なフレームをスキップ

            # bboxがNoneの場合はスキップ
            if bbox is None:
                continue  # 無効なbboxをスキップ

            # フレームをuint8形式にキャストし、(h, w, 3)の形に変換
            img_uint = np.transpose(frame.numpy(), (1, 2, 0)).astype('uint8').copy()  # .copy() を追加して連続メモリ化
            
            # bboxの描画
            for box in bbox:
                try:
                    x, y, w, h, cls = box['x'], box['y'], box['w'], box['h'], box['class_id']
                    if w > 0 and h > 0:  # 有効なボックスのみ描画
                        start_point = (int(x), int(y))
                        end_point = (int(x + w), int(y + h))
                        
                        # 黄色で太さ2の矩形描画
                        cv2.rectangle(img_uint, start_point, end_point, (0, 255, 255), 2)  
                        cv2.putText(img_uint, f"Cls: {int(cls)}", (int(x), int(y) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error drawing rectangle at index {index}, time step {t}, box {box}: {e}")

            # フレームを動画に追加
            video_writer.write(img_uint)
    
    # 動画ファイルを保存
    video_writer.release()
    print(f"動画が保存されました: {output_file}")



def save_sequence_as_video_from_dataloader(dataloader: DataLoader, t_ms: int, output_file: str):
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

        # バッチ内の処理
        for b in range(events.shape[0]):
            for t in range(events.shape[1]):
                if mask is not None and mask[b, t].item() == 0:
                    continue

                # フレーム変換と描画
                frame = events[b, t]
                img_uint = np.transpose(frame.numpy(), (1, 2, 0)).astype('uint8').copy()
                if labels.shape[2] > 0:
                    for box in labels[b, t]:
                        x, y, w, h, cls = box[:5]
                        if w > 0 and h > 0:
                            start_point = (int(x), int(y))
                            end_point = (int(x + w), int(y + h))
                            cv2.rectangle(img_uint, start_point, end_point, (0, 255, 255), 2)
                            cv2.putText(img_uint, f"Cls: {int(cls)}", (int(x), int(y) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # フレーム書き込み
                video_writer.write(img_uint)

    video_writer.release()
    print(f"動画が保存されました: {output_file}")

