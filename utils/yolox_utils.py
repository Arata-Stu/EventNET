import torch
import torchvision

def to_yolox(tensor, mode='train'):
    """
    テンソルから指定の情報を抽出してフォーマットを調整する関数。
    
    Args:
    - tensor (torch.Tensor): torch.Size([B, L, num_obj, info]) の形状のテンソル
    - mode (str): 'train', 'test', または 'val'。データの形式を決定するための引数。
    
    Returns:
    - torch.Tensor: 指定の形に変換されたテンソル
    """
    # shapeの確認
    B, L, num_obj, info = tensor.shape
    assert info == 8, "入力テンソルのinfoは8でなければなりません"

    # 各情報の抽出（timestamp, class_confidence, track_id は無視）
    x = tensor[:, :, :, 1:2]  # x
    y = tensor[:, :, :, 2:3]  # y
    w = tensor[:, :, :, 3:4]  # width
    h = tensor[:, :, :, 4:5]  # height
    cls = tensor[:, :, :, 5:6]  # class_id

    if mode == 'train':
        # trainモードの時は (cls, cx, cy, w, h) のフォーマット
        cx = x + w / 2  # 中心 x 座標
        cy = y + h / 2  # 中心 y 座標
        output = torch.cat([cls, cx, cy, w, h], dim=-1)
        
    elif mode in ['test', 'val']:
        # testまたはvalモードの時は (x, y, w, h, cls) のフォーマット
        output = torch.cat([x, y, w, h, cls], dim=-1)
    
    else:
        raise ValueError("modeは'train', 'test', 'val'のいずれかである必要があります")

    return output

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output
