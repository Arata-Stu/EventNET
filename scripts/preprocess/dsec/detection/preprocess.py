import os
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm
import yaml
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

    # オンイベントを赤、オフイベントを青に割り当て
    frame[slice_events['y'][off_events], slice_events['x'][off_events]] = np.array([0, 0, 255], dtype=np.uint8)
    frame[slice_events['y'][on_events], slice_events['x'][on_events]] = np.array([255, 0, 0], dtype=np.uint8)

    return frame

# 各シーケンスの処理
def process_sequence(args):
    data_dir, output_dir, seq, tau_ms, delta_t_ms, frame_shape = args

    # tau_ms と delta_t_ms をマイクロ秒に変換
    tau_us = tau_ms * 1000
    delta_t_us = delta_t_ms * 1000

    seq_output_dir = os.path.join(output_dir, seq, f"tau={tau_ms}_dt={delta_t_ms}")
    
    print(f"Processing sequence: {seq}")
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

    os.makedirs(seq_output_dir, exist_ok=True)
    start_time = events['t'][0]
    end_time = events['t'][-1]
    window_starts = np.arange(start_time, end_time, tau_us)

    for i, start in enumerate(window_starts[:-1]):
        end = window_starts[i + 1]
        start_range = max(start - delta_t_us, start_time)
        start_idx = np.searchsorted(events['t'], start_range)
        end_idx = np.searchsorted(events['t'], end)

        timestamp_str = f"{int(start)}_to_{int(end)}"
        event_file_name = os.path.join(seq_output_dir, f"{timestamp_str}_event.npz")
        label_file_name = os.path.join(seq_output_dir, f"{timestamp_str}_label.npz")

        if os.path.exists(event_file_name) and os.path.exists(label_file_name):
            continue

        # イベントスライスを生成
        slice_events = {
            't': events['t'][start_idx:end_idx],
            'x': events['x'][start_idx:end_idx],
            'y': events['y'][start_idx:end_idx],
            'p': events['p'][start_idx:end_idx],
        }

        # イベントフレームを生成
        event_frame = create_event_frame(slice_events, frame_shape)

        # イベントフレームを保存
        np.savez_compressed(event_file_name, events=event_frame)

        if detections.size > 0:
            # タイムウィンドウ内の検出データを取得
            det_mask = (detections['t'] >= start_range) & (detections['t'] < end)
            slice_detections = detections[det_mask]

            labels = []
            unique_track_ids = np.unique(slice_detections['track_id'])
            
            for track_id in unique_track_ids:
                # 同じtrack_idを持つ検出データをフィルタし、タイムスタンプが最大のものを取得
                track_detections = slice_detections[slice_detections['track_id'] == track_id]
                latest_detection = track_detections[np.argmax(track_detections['t'])]  # タイムスタンプが最大のものを取得

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

            # ラベルを圧縮保存
            np.savez_compressed(label_file_name, labels=labels)
            has_label = True
        else:
            labels = []
            has_label = False
            label_file_name = None  # ラベルがない場合

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

# メイン処理
def main(config):
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    num_processors = config.get("num_processors", cpu_count())
    tau_ms = config["tau_ms"]
    delta_t_ms = config["delta_t_ms"]
    frame_shape = tuple(config["frame_shape"])

    sequences = [seq for seq in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, seq))]
    os.makedirs(output_dir, exist_ok=True)

    with tqdm(total=len(sequences), desc="Processing sequences") as pbar:
        with get_context('spawn').Pool(processes=num_processors) as pool:
            args_list = [(input_dir, output_dir, seq, tau_ms, delta_t_ms, frame_shape) for seq in sequences]
            for _ in pool.imap_unordered(process_sequence, args_list):
                pbar.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process event data with configuration file")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    start_time = time.time()

    main(config)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"逐次処理の完了時間: {elapsed_time:.2f} 秒")
