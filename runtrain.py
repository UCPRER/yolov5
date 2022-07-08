import argparse
from pathlib import Path
import yaml
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', default='', help='where to store temp files')
    parser.add_argument('--log-enable', action='store_true', help='output log to file')
    parser.add_argument('--label-type', default='coco', choices=['voc', 'coco'], help='dataset labeling method')
    parser.add_argument('--train-list', default='', help='path of training set list')
    parser.add_argument('--val-list', default='', help='path of validation set list')
    parser.add_argument('--train-annotation', default='', help='path of training set annotation')
    parser.add_argument('--val-annotation', default='', help='path of validation set annotation')
    parser.add_argument('--config', default='', help='path of training config(Hyperparameters)')
    parser.add_argument('--device', default='', help='cpu or cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()
    return args


def generate_coco_yaml(root, train_list, val_list, train_label, val_label):
    obj = {
        "train": str(train_list),
        "val": str(val_list),
        "train_label": str(train_label),
        "val_label": str(val_label),
        "nc": 80,
        "names": [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ],
    }
    with open(root / "coco.yaml", "w") as f:
        yaml.safe_dump(data=obj, stream=f)


def main():
    args = get_args()
    assert args.work_dir, "work-dir is required"
    root = Path(args.work_dir)

    if args.log_enable:
        out_file = open(root / "out.txt", "w")
        sys.stdout = out_file
        sys.stderr = out_file
    import train
    if args.label_type == 'coco':
        assert args.train_list, "train-list is required"
        assert args.val_list, "val-list is required"
        assert args.train_annotation, "train-annotation is required"
        assert args.val_annotation, "val-annatation is required"

        # 格式转换
        from converter import coco2yolo
        coco2yolo(json_path=args.train_annotation, save_path=root / "labels/train")
        coco2yolo(json_path=args.val_annotation, save_path=root / "labels/val")

        generate_coco_yaml(root, args.train_list, args.val_list, root / "labels/train", root / "labels/val")

        coco = True if args.label_type == "coco" else False
        train.run(data=root / "coco.yaml", imgsz=640, batch_size=96, epochs=300, cfg='yolov5s.yaml', is_coco=coco, project=root,
                  name="result", device=args.device)


if __name__ == "__main__":
    main()
