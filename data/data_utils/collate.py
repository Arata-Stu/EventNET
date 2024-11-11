import torch
import numpy as np

def custom_collate_fn(batch):
    # batch内のサンプルに`labels`が空かどうか確認し、空であれば0を使用
    sequence_length = max(len(sample['labels']) for sample in batch if sample['labels']) or 0
    max_bboxes = max(
        (max(len(time_step) for time_step in sample['labels'] if time_step is not None) 
         for sample in batch if sample['labels']), default=0
    )

    # 各サンプルのデータを収集
    event_frames = [torch.stack([frame.clone().detach() for frame in sample['events']]) for sample in batch]
    labels = [sample['labels'] for sample in batch]
    is_start_sequence = [sample['is_start_sequence'] for sample in batch]
    masks = [sample['mask'].clone().detach().to(dtype=torch.int64) for sample in batch]
    timestamps = [sample['timestamps'] for sample in batch]

    # `labels`をバッチサイズ、シーケンス長、最大bbox数、属性数+1に合わせてテンソル化
    padded_labels = []
    for i, label in enumerate(labels):
        sequence_labels = []
        for time_step in range(sequence_length):
            if time_step < len(label) and label[time_step] is not None:
                bboxes = label[time_step]
                bbox_tensor = torch.zeros((max_bboxes, 8))  # 8は属性数（timestamp, x, y, w, h, class_id, class_confidence, track_id）
                for j, bbox in enumerate(bboxes[:max_bboxes]):
                    bbox_tensor[j] = torch.tensor([
                        timestamps[i][time_step],  # タイムスタンプを先頭に追加
                        bbox['x'], bbox['y'], bbox['w'], bbox['h'],
                        bbox['class_id'], bbox['class_confidence'], bbox['track_id']
                    ])
                sequence_labels.append(bbox_tensor)
            else:
                # bboxがない場合はゼロテンソルを追加
                sequence_labels.append(torch.zeros((max_bboxes, 8)))
        padded_labels.append(torch.stack(sequence_labels))

    # `padded_labels`をテンソル化してバッチサイズでスタック
    padded_labels = torch.stack(padded_labels)

    # event_framesとmasksをテンソルに変換
    event_frames = torch.stack(event_frames)
    masks = torch.stack(masks)

    return {
        'events': event_frames,
        'labels': padded_labels,
        'timestamps': torch.stack([ts.clone().detach() for ts in timestamps]),  
        'is_start_sequence': is_start_sequence,
        'mask': masks
    }
