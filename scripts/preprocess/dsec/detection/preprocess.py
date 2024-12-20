import os
import h5py
import hdf5plugin
from typing import Tuple
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf  # omegaconfをインポート
from multiprocessing import Pool, cpu_count, get_context
import argparse
import json
import time

# イベントフレームの生成関数
def create_event_frame(slice_events, frame_shape):
    height, width = frame_shape
    frame = np.ones((height, width, 3), dtype=np.uint8) * 114  # 背景をグレーで初期化

    # オン・オフイベントのマスクを作成
    off_events = (slice_events['p'] == 0)
    on_events = (slice_events['p'] == 1)

    # クリップ処理を追加
    x_clipped = np.clip(slice_events['x'], 0, width - 1)
    y_clipped = np.clip(slice_events['y'], 0, height - 1)

    # オンイベントを赤、オフイベントを青に割り当て
    frame[y_clipped[off_events], x_clipped[off_events]] = np.array([0, 0, 255], dtype=np.uint8)
    frame[y_clipped[on_events], x_clipped[on_events]] = np.array([255, 0, 0], dtype=np.uint8)

    return frame

def create_event_histogram(slice_events, frame_shape: Tuple[int, int], bins: int) -> np.ndarray:
    """イベントデータ（辞書形式）からイベントヒストグラムを生成"""
    height, width = frame_shape
    representation = np.zeros((2, bins, height, width), dtype=np.uint8)  # 2はpolarityの数

    # イベントが空の場合
    if len(slice_events['x']) == 0:
        return representation.reshape(-1, height, width)

    # 正規化した時間インデックスの計算
    t0 = slice_events['t'][0]
    t1 = slice_events['t'][-1]
    t_norm = (slice_events['t'] - t0) / max((t1 - t0), 1)
    t_idx = np.clip((t_norm * bins).astype(int), 0, bins - 1)

    for p in [0, 1]:  # Polarityで分けて処理
        mask = (slice_events['p'] == p)
        bin_indices = t_idx[mask]

        # クリップ処理を追加
        x_clipped = np.clip(slice_events['x'][mask], 0, width - 1)
        y_clipped = np.clip(slice_events['y'][mask], 0, height - 1)

        for b, xi, yi in zip(bin_indices, x_clipped, y_clipped):
            representation[p, b, yi, xi] += 1

    return representation.reshape(-1, height, width)


# 各シーケンスの処理
def process_sequence(args):
    data_dir, output_dir, representation_type, bins, seq, tau_ms, delta_t_ms, frame_shape = args

    # tau_ms と delta_t_ms をマイクロ秒に変換
    tau_us = tau_ms * 1000
    delta_t_us = delta_t_ms * 1000

    # 出力ディレクトリを `tau` と `delta_t` の組み合わせで分ける
    tau_delta_dir = f"tau={tau_ms}_dt={delta_t_ms}"
    seq_output_dir = os.path.join(output_dir, tau_delta_dir, seq)
    os.makedirs(seq_output_dir, exist_ok=True)
    
    print(f"Processing sequence: {seq} with tau={tau_ms} and delta_t={delta_t_ms}")
    
    # イベントデータと検出データのパス
    event_path = os.path.join(data_dir, seq, 'events', 'left', 'events.h5')
    detection_path = os.path.join(data_dir, seq, 'object_detections', 'left', 'tracks.npy')

    # インデックスリストの初期化
    index_list = []

    # イベントデータの読み込み
    with h5py.File(event_path, 'r') as f:
        t_offset = f['t_offset'][()]
        events = {
            't': f['events']['t'][:] + t_offset,  # オフセットを加算（マイクロ秒）
            'x': f['events']['x'][:],
            'y': f['events']['y'][:],
            'p': f['events']['p'][:]
        }

    # オブジェクト検出データの読み込み（存在しない場合は空の配列を用意）
    if os.path.exists(detection_path):
        detections = np.load(detection_path)
    else:
        detections = np.array([], dtype=[
            ('t', 'float64'),
            ('x', 'float64'),
            ('y', 'float64'),
            ('w', 'float64'),
            ('h', 'float64'),
            ('class_id', 'int32'),
            ('class_confidence', 'float64'),
            ('track_id', 'int32')
        ])

    start_time = events['t'][0]
    end_time = events['t'][-1]
    window_starts = np.arange(start_time, end_time, tau_us)

    for i, start in enumerate(window_starts[:-1]):
        end = window_starts[i + 1]
        start_range = max(start - delta_t_us, start_time)
        start_idx = np.searchsorted(events['t'], start_range)
        end_idx = np.searchsorted(events['t'], end)

        timestamp_str = f"{int(start)}_to_{int(end)}"
        event_file_name = f"{timestamp_str}_event.npz"
        event_save_name = os.path.join(seq_output_dir, event_file_name)
        label_file_name = f"{timestamp_str}_label.npz"
        label_save_name = os.path.join(seq_output_dir, label_file_name)

        if os.path.exists(event_save_name) and os.path.exists(label_save_name):
            continue

        # イベントスライスを生成
        slice_events = {
            't': events['t'][start_idx:end_idx],
            'x': events['x'][start_idx:end_idx],
            'y': events['y'][start_idx:end_idx],
            'p': events['p'][start_idx:end_idx],
        }

        # イベントフレームを作成
        if representation_type == "frame":
            event_frame = create_event_frame(slice_events, frame_shape)
        elif representation_type == "histogram":
            event_frame = create_event_histogram(slice_events, frame_shape, bins)
        else:
            raise ValueError(f"Unknown representation type: {representation_type}")

        # イベントフレームを保存
        np.savez_compressed(event_save_name, events=event_frame)

        # ラベル保存処理
        if detections.size > 0:
            det_mask = (detections['t'] >= start_range) & (detections['t'] < end)
            slice_detections = detections[det_mask]
            labels = []

            for track_id in np.unique(slice_detections['track_id']):
                track_detections = slice_detections[slice_detections['track_id'] == track_id]
                latest_detection = track_detections[np.argmax(track_detections['t'])]
                labels.append({
                    't': latest_detection['t'],
                    'x': latest_detection['x'],
                    'y': latest_detection['y'],
                    'w': latest_detection['w'],
                    'h': latest_detection['h'],
                    'class_id': latest_detection['class_id'],
                    'class_confidence': latest_detection['class_confidence'],
                    'track_id': latest_detection['track_id']
                })

            np.savez_compressed(label_save_name, labels=labels)
            has_label = True
        else:
            labels = []
            has_label = False
            label_file_name = None

        # インデックスにエントリ追加
        index_entry = {
            'event_file': event_file_name,
            'label_file': label_file_name,
            'timestamp': (int(start), int(end))
        }
        index_list.append(index_entry)

    # インデックスファイルをJSON形式で保存
    index_file_path = os.path.join(seq_output_dir, 'index.json')
    with open(index_file_path, 'w') as index_file:
        json.dump(index_list, index_file)

    print(f"Completed processing sequence: {seq}")


def main(config):
    input_dir = config.input_dir
    output_dir = config.output_dir
    representation_type = config.get("representation_type", "frame")
    print('representation_type', representation_type)
    bins = config.get("bins", 10)
    num_processors = int(config.get("num_processors", cpu_count()))
    tau_ms = config.tau_ms
    delta_t_ms = config.delta_t_ms
    frame_shape = tuple(config.frame_shape)

    sequences = [seq for seq in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, seq))]
    os.makedirs(output_dir, exist_ok=True)

    with tqdm(total=len(sequences), desc="Processing sequences") as pbar:
        with get_context('spawn').Pool(processes=num_processors) as pool:
            args_list = [(input_dir, output_dir, representation_type, bins, seq, tau_ms, delta_t_ms, frame_shape) for seq in sequences]
            for _ in pool.imap_unordered(process_sequence, args_list):
                pbar.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process event data with configuration file")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    # omegaconfを使用して設定ファイルを読み込む
    config = OmegaConf.load(args.config)

    start_time = time.time()

    main(config)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"逐次処理の完了時間: {elapsed_time:.2f} 秒")