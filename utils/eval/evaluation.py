import numpy as np
import os
import contextlib
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluation(Gt, Dt, num_data):

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):

        out_keys = ('AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L')
        out_dict = {k: 0.0 for k in out_keys}

        if len(Dt) == 0:
            return out_dict
        
        coco_gt = COCO()
        coco_gt.dataset = Gt

        coco_gt.createIndex()
        coco_pred = coco_gt.loadRes(Dt)

        coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
        coco_eval.params.imgIds = np.arange(1, num_data + 1, dtype=int)
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        # info: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
        coco_eval.summarize()
        for idx, key in enumerate(out_keys):
            out_dict[key] = coco_eval.stats[idx]

    return out_dict


def to_coco_format(gts, detections, categories, height=640, width=640):
    """
    utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append(
            {"date_captured": "2017",
             "file_name": "n.a",
             "id": im_id,
             "license": 1,
             "url": "",
             "height": height,
             "width": width})

        # Skip if gt or pred is None
        if gt is None or pred is None:
            continue

        for bbox in gt:
            x, y, w, h = bbox[:4].tolist()  # Extract bounding box coordinates
            if w <= 0 or h <= 0:
                continue  # 幅または高さが0以下なら無視 0 paddingを無視する
            area = w * h

            annotation = {
                "area": float(area),  # Area of the bounding box
                "iscrowd": False,  # Set 'iscrowd' to False for non-crowd annotations
                "image_id": im_id,  # ID of the image this annotation belongs to
                "bbox": [x, y, w, h],  # COCO format expects [x, y, width, height]
                "category_id": int(bbox[4]) + 1,  # Class ID from bbox, assuming it's in index 4
                "id": len(annotations) + 1  # Unique ID for each annotation
            }
            annotations.append(annotation)

        for bbox in pred:
            # Extract coordinates, confidence, and class id
            x1, y1, w, h = bbox[:4].tolist()
            score = bbox[4].item()  # Confidence score
            class_conf = bbox[5].item()  # Class confidence (optional)
            class_id = int(bbox[6].item())  # Class id

            image_result = {
                'image_id': im_id,
                'category_id': class_id + 1,
                'score': score,  # Optionally, you can combine with class_conf if needed
                'bbox': [x1, y1, w, h],
            }
            results.append(image_result)

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}
    return dataset, results
