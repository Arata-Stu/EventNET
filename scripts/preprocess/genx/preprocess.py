import os
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, get_context
from typing import Tuple
import yaml
import argparse
import re
import cv2  # OpenCVを使用して解像度変更
import json
import time

# BBOXデータタイプ
BBOX_DTYPE = np.dtype({
    'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
    'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
    'offsets': [0, 8, 12, 16, 20, 24, 28, 32],
    'itemsize': 40
})

def find_event_and_bbox_files(sequence_dir, mode):
    """指定されたディレクトリ内でイベントデータとBBOXファイルを検索し、`gen1`と`gen4`で処理を分ける"""
    files = os.listdir(sequence_dir)
    event_file = None
    bbox_file = None
    if mode == 'gen1':
        event_pattern = r'_td\.dat\.h5$'
        bbox_pattern = r'_bbox\.npy$'
    elif mode == 'gen4':
        event_pattern = r'_td\.h5$'
        bbox_pattern = r'_bbox\.npy$'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for file in files:
        if re.search(event_pattern, file):
            event_file = file
        elif re.search(bbox_pattern, file):
            bbox_file = file

    if not event_file or not bbox_file:
        raise FileNotFoundError(f"イベントまたはBBOXファイルが見つかりません: {sequence_dir}")
    return event_file, bbox_file

def create_event_frame(slice_events, frame_shape, downsample=False):
    height, width = frame_shape
    frame = np.ones((height, width, 3), dtype=np.uint8) * 114  # グレーバックグラウンド

    # オン・オフイベントのマスクを作成
    off_events = (slice_events['p'] == -1)
    on_events = (slice_events['p'] == 1)

    # オンイベントを赤、オフイベントを青に割り当て
    frame[slice_events['y'][off_events], slice_events['x'][off_events]] = np.array([0, 0, 255], dtype=np.uint8)
    frame[slice_events['y'][on_events], slice_events['x'][on_events]] = np.array([255, 0, 0], dtype=np.uint8)

    # downsampleがTrueの場合、解像度を半分にする
    if downsample:
        frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        
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
        x_indices = slice_events['x'][mask]
        y_indices = slice_events['y'][mask]

        for b, xi, yi in zip(bin_indices, x_indices, y_indices):
            if 0 <= yi < height and 0 <= xi < width:
                representation[p, b, yi, xi] += 1

    return representation.reshape(-1, height, width)


def conservative_bbox_filter(labels: np.ndarray) -> np.ndarray:
    min_box_side = 5
    w_lbl = labels['w']
    h_lbl = labels['h']
    side_ok = (w_lbl >= min_box_side) & (h_lbl >= min_box_side)
    return labels[side_ok]

def remove_faulty_huge_bbox_filter(labels: np.ndarray, dataset_type: str) -> np.ndarray:
    assert dataset_type in {'gen1', 'gen4'}
    max_width = (9 * (1280 if dataset_type == 'gen4' else 304)) // 10
    side_ok = (labels['w'] <= max_width)
    return labels[side_ok]

def crop_to_fov_filter(labels: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
    frame_height, frame_width = frame_shape
    x_left = labels['x']
    y_top = labels['y']
    x_right = x_left + labels['w']
    y_bottom = y_top + labels['h']
    
    x_left_cropped = np.clip(x_left, 0, frame_width - 1)
    y_top_cropped = np.clip(y_top, 0, frame_height - 1)
    x_right_cropped = np.clip(x_right, 0, frame_width - 1)
    y_bottom_cropped = np.clip(y_bottom, 0, frame_height - 1)
    
    labels['x'] = x_left_cropped
    labels['y'] = y_top_cropped
    labels['w'] = x_right_cropped - x_left_cropped
    labels['h'] = y_bottom_cropped - y_top_cropped
    
    keep = (labels['w'] > 0) & (labels['h'] > 0)
    return labels[keep]


def prophesee_bbox_filter(labels: np.ndarray, dataset_type: str) -> np.ndarray:
    assert dataset_type in {'gen1', 'gen4'}

    # デフォルト値
    min_box_diag = 60 if dataset_type == 'gen4' else 30
    min_box_side = 20 if dataset_type == 'gen4' else 10

    w_lbl = labels['w']
    h_lbl = labels['h']

    diag_ok = w_lbl ** 2 + h_lbl ** 2 >= min_box_diag ** 2
    side_ok = (w_lbl >= min_box_side) & (h_lbl >= min_box_side)
    keep = diag_ok & side_ok
    labels = labels[keep]
    return labels

def apply_filters(labels: np.ndarray, dataset_type: str, frame_shape: Tuple[int, int]) -> np.ndarray:
    labels = prophesee_bbox_filter(labels, dataset_type)
    labels = conservative_bbox_filter(labels)
    labels = remove_faulty_huge_bbox_filter(labels, dataset_type)
    labels = crop_to_fov_filter(labels, frame_shape)
    return labels

def process_sequence(args):
    data_dir, output_dir, representation_type, bins, split, seq, tau_ms, delta_t_ms, frame_shape, mode, downsample = args
    print(f"Processing : {seq} in split: {split}")


    tau_us = tau_ms * 1000
    delta_t_us = delta_t_ms * 1000

    seq_output_dir = os.path.join(output_dir, split, seq, f"tau={tau_ms}_dt={delta_t_ms}")
    os.makedirs(seq_output_dir, exist_ok=True)

    sequence_dir = os.path.join(data_dir, split, seq)
    event_file, bbox_file = find_event_and_bbox_files(sequence_dir, mode)

    event_path = os.path.join(sequence_dir, event_file)
    bbox_path = os.path.join(sequence_dir, bbox_file)

    # インデックスリストの初期化
    index_list = []

    with h5py.File(event_path, 'r') as f:
        events = {
            't': f['events']['t'][:],
            'x': f['events']['x'][:],
            'y': f['events']['y'][:],
            'p': f['events']['p'][:]
        }

    if os.path.exists(bbox_path):
        detections = np.load(bbox_path)
    else:
        detections = np.array([], dtype=BBOX_DTYPE)

    events['t'] = events['t'].astype(np.float64)
    detections['t'] = detections['t'].astype(np.float64)

    start_time = max(events['t'][0], 10000)
    window_times = np.arange(start_time, events['t'][-1], tau_us)

    for t in window_times:
        data_start = t - delta_t_us
        data_end = t

        start_idx = np.searchsorted(events['t'], data_start)
        end_idx = np.searchsorted(events['t'], data_end)

        slice_events = {
            't': events['t'][start_idx:end_idx],
            'x': events['x'][start_idx:end_idx],
            'y': events['y'][start_idx:end_idx],
            'p': events['p'][start_idx:end_idx],
        }

        if detections.size > 0:
            label_mask = (detections['t'] >= (t - tau_us / 2)) & (detections['t'] < (t + tau_us / 2))
            slice_detections = detections[label_mask]

            unique_detections = {}
            for det in slice_detections:
                track_id = det['track_id']
                if track_id not in unique_detections or det['t'] > unique_detections[track_id]['t']:
                    unique_detections[track_id] = det

            labels_array = np.array([
                (
                    label['t'],
                    label['x'],
                    label['y'],
                    label['w'],
                    label['h'],
                    label['class_id'],
                    label['track_id'],
                    label['class_confidence']
                ) for label in unique_detections.values()
            ], dtype=BBOX_DTYPE)

            # ここで複数フィルタを適用
            labels_filtered = apply_filters(labels_array, mode, frame_shape)

            labels = [
                {
                    't': label['t'],
                    'x': label['x'] // 2 if downsample else label['x'],
                    'y': label['y'] // 2 if downsample else label['y'],
                    'w': label['w'] // 2 if downsample else label['w'],
                    'h': label['h'] // 2 if downsample else label['h'],
                    'class_id': label['class_id'],
                    'class_confidence': label['class_confidence'],
                    'track_id': label['track_id']
                }
                for label in labels_filtered
            ]
        else:
            labels = []


        # イベントフレームを作成
        if representation_type == "frame":
            event_frame = create_event_frame(slice_events, frame_shape, downsample=downsample)
        elif representation_type == "histogram":
            event_frame = create_event_histogram(slice_events, frame_shape, bins)
        else:
            raise ValueError(f"Unknown representation type: {representation_type}")


        # ファイル名の定義
        timestamp_str = f"{int(data_start)}_to_{int(data_end)}"
        event_file_name = os.path.join(seq_output_dir, f"{timestamp_str}_event.npz")
        label_file_name = os.path.join(seq_output_dir, f"{timestamp_str}_label.npz")

        # イベントフレームを圧縮形式で保存
        if not os.path.exists(event_file_name):
            np.savez_compressed(event_file_name, events=event_frame)

        # ラベルが存在する場合のみ圧縮形式で保存
        if labels:
            if not os.path.exists(label_file_name):
                np.savez_compressed(label_file_name, labels=labels)
            has_label = True
        else:
            has_label = False
            label_file_name = None  # ラベルファイルがない場合はNoneに設定


        # インデックスリストに追加
        index_entry = {
            'event_file': event_file_name,
            'label_file': label_file_name,
            'timestamp': (int(data_start), int(data_end))
        }
        index_list.append(index_entry)

    # インデックスファイルをJSON形式で保存
    index_file_path = os.path.join(seq_output_dir, 'index.json')
    with open(index_file_path, 'w') as index_file:
        json.dump(index_list, index_file)

    print(f"Completed processing sequence: {seq} in split: {split}")

def main(config):
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    num_processors = config.get("num_processors", cpu_count())
    representation_type = config.get("representation_type", "frame")
    print('representation_type', representation_type)
    bins = config.get("bins", 10)
    tau_ms = config["tau_ms"]
    delta_t_ms = config["delta_t_ms"]
    frame_shape = tuple(config["frame_shape"])
    mode = config.get("mode", "gen1")
    downsample = config.get("downsample", False)

    splits = ['train', 'test', 'val']
    sequences = [(split, seq) for split in splits for seq in os.listdir(os.path.join(input_dir, split)) if os.path.isdir(os.path.join(input_dir, split, seq))]

    with tqdm(total=len(sequences), desc="Processing sequences") as pbar:
        with get_context('spawn').Pool(processes=num_processors) as pool:
            args_list = [(input_dir, output_dir, representation_type, bins, split, seq, tau_ms, delta_t_ms, frame_shape, mode, downsample) for split, seq in sequences]
            for _ in pool.imap_unordered(process_sequence, args_list):
                pbar.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process gen1/gen4 dataset with configuration file")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    start_time = time.time()

    main(config)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"逐次処理の完了時間: {elapsed_time:.2f} 秒")