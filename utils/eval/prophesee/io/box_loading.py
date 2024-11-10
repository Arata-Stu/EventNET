# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Defines some tools to handle events.
In particular :
    -> defines events' types
    -> defines functions to read events from binary .dat files using numpy
    -> defines functions to write events to binary .dat files using numpy
"""

from __future__ import print_function

from typing import List, Tuple, Optional

import numpy as np
import torch as th


BBOX_DTYPE = np.dtype({'names':['t','x','y','w','h','class_id','track_id','class_confidence'], 'formats':['<i8','<f4','<f4','<f4','<f4','<u4','<u4','<f4'], 'offsets':[0,8,12,16,20,24,28,32], 'itemsize':40})
YOLOX_PRED_PROCESSED = List[Optional[th.Tensor]]


def reformat_boxes(boxes):
    """ReFormat boxes according to new rule
    This allows to be backward-compatible with imerit annotation.
        't' = 'ts'
        'class_confidence' = 'confidence'
    """
    if 't' not in boxes.dtype.names or 'class_confidence' not in boxes.dtype.names:
        new = np.zeros((len(boxes),), dtype=BBOX_DTYPE) 
        for name in boxes.dtype.names:
            if name == 'ts':
                new['t'] = boxes[name]
            elif name == 'confidence':
                new['class_confidence'] = boxes[name]
            else:
                new[name] = boxes[name]
        return new
    else:
        return boxes

def process_labels(labels: th.Tensor, time: int) -> np.ndarray:
    """
    Process labels from a torch.Tensor into a numpy array with BBOX_DTYPE.

    Args:
        labels: torch.Tensor of shape (num_labels, 5), where each label is (x, y, w, h, class_id)
        time: The timestamp associated with these labels.

    Returns:
        A numpy array of labels with dtype BBOX_DTYPE.
    """
    num_labels = labels.shape[0]
    labels_proph = np.zeros((num_labels,), dtype=BBOX_DTYPE)
    labels_np = labels.detach().cpu().numpy()

    labels_proph['t'] = np.full((num_labels,), time, dtype=BBOX_DTYPE['t'])
    labels_proph['x'] = labels_np[:, 0].astype(BBOX_DTYPE['x'])
    labels_proph['y'] = labels_np[:, 1].astype(BBOX_DTYPE['y'])
    labels_proph['w'] = labels_np[:, 2].astype(BBOX_DTYPE['w'])
    labels_proph['h'] = labels_np[:, 3].astype(BBOX_DTYPE['h'])
    labels_proph['class_id'] = labels_np[:, 4].astype(BBOX_DTYPE['class_id'])
    labels_proph['class_confidence'] = np.zeros((num_labels,), dtype=BBOX_DTYPE['class_confidence'])
    labels_proph['track_id'] = np.zeros((num_labels,), dtype=BBOX_DTYPE['track_id'])

    return labels_proph


def process_yolox_preds(yolox_preds: Optional[th.Tensor], time: int) -> np.ndarray:
    """
    Process YOLOX predictions into a numpy array with BBOX_DTYPE.

    Args:
        yolox_preds: A torch.Tensor or numpy array of shape (num_preds, 7).
                     Each prediction is (x1, y1, x2, y2, obj_conf, class_conf, class_pred).
        time: The timestamp associated with these predictions.

    Returns:
        A numpy array of predictions with dtype BBOX_DTYPE.
    """
    if yolox_preds is None or len(yolox_preds) == 0:
        return np.zeros((0,), dtype=BBOX_DTYPE)

    if isinstance(yolox_preds, th.Tensor):
        yolox_preds = yolox_preds.detach().cpu().numpy()

    num_pred = yolox_preds.shape[0]
    yolox_pred_proph = np.zeros((num_pred,), dtype=BBOX_DTYPE)

    yolox_pred_proph['t'] = np.full((num_pred,), time, dtype=BBOX_DTYPE['t'])
    yolox_pred_proph['x'] = yolox_preds[:, 0].astype(BBOX_DTYPE['x'])
    yolox_pred_proph['y'] = yolox_preds[:, 1].astype(BBOX_DTYPE['y'])
    yolox_pred_proph['w'] = (yolox_preds[:, 2] - yolox_preds[:, 0]).astype(BBOX_DTYPE['w'])
    yolox_pred_proph['h'] = (yolox_preds[:, 3] - yolox_preds[:, 1]).astype(BBOX_DTYPE['h'])
    yolox_pred_proph['class_id'] = yolox_preds[:, 6].astype(BBOX_DTYPE['class_id'])
    yolox_pred_proph['class_confidence'] = yolox_preds[:, 5].astype(BBOX_DTYPE['class_confidence'])
    yolox_pred_proph['track_id'] = np.zeros((num_pred,), dtype=BBOX_DTYPE['track_id'])

    return yolox_pred_proph


def to_prophesee(
    loaded_label_tensor: th.Tensor,
    label_timestamps: th.Tensor,
    yolox_pred_list: YOLOX_PRED_PROCESSED
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Convert labels and YOLOX predictions into Prophesee format.

    Args:
        loaded_label_tensor: torch.Tensor of shape (batch_size, num_labels, 5)
        label_timestamps: torch.Tensor of shape (batch_size,)
        yolox_pred_list: List of YOLOX prediction tensors or arrays, one per timestamp.

    Returns:
        A tuple containing:
            - List of numpy arrays with processed labels for each timestamp.
            - List of numpy arrays with processed YOLOX predictions for each timestamp.
    """
    batch_size = loaded_label_tensor.shape[0]
    assert batch_size == label_timestamps.shape[0] == len(yolox_pred_list), \
        "All input lists must have the same length."

    loaded_label_list_proph = []
    yolox_pred_list_proph = []

    for i in range(batch_size):
        labels = loaded_label_tensor[i]  # Shape: (num_labels, 5)
        time = int(label_timestamps[i].item())
        yolox_preds = yolox_pred_list[i]  # Could be None or a tensor of shape (num_preds, 7)

        # Process labels for the current timestamp
        labels_proph = process_labels(labels, time)
        loaded_label_list_proph.append(labels_proph)

        # Process YOLOX predictions for the current timestamp
        yolox_pred_proph = process_yolox_preds(yolox_preds, time)
        yolox_pred_list_proph.append(yolox_pred_proph)

    return loaded_label_list_proph, yolox_pred_list_proph