import argparse
from pathlib import Path
import yaml
import sys
import json
from pycocotools.coco import COCO
import logging
from utils.general import suppress_stdout


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="", help="where to store temp files")
    parser.add_argument("--log-enable", action="store_true", help="output log to file")
    parser.add_argument("--label-type", default="coco", choices=["voc", "coco"], help="dataset labeling method")
    parser.add_argument("--train-list", default="", help="path of training set list")
    parser.add_argument("--val-list", default="", help="path of validation set list")
    parser.add_argument("--train-annotation", default="", help="path of training set annotation")
    parser.add_argument("--val-annotation", default="", help="path of validation set annotation")
    parser.add_argument("--config", default="", help="path of training config(Hyperparameters, JSON file)")
    parser.add_argument("--device", default="", help="cpu or cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("--development", action="store_true", help="development mode")

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def generate_coco_yaml(root, train_list, val_list, train_label, val_label, json_path, verbose=False):
    if verbose:
        coco = COCO(json_path)
    else:
        with suppress_stdout():
            coco = COCO(json_path)
    catids = coco.getCatIds()
    cats = coco.loadCats(catids)

    obj = {
        "train": str(train_list),
        "val": str(val_list),
        "train_label": str(train_label),
        "val_label": str(val_label),
        "nc": len(catids),
        "names": [cat["name"] for cat in cats],
    }
    with open(root / "coco.yaml", "w") as f:
        yaml.safe_dump(data=obj, stream=f)


def main():
    logger = logging.getLogger(__name__)
    args, unknown_args = get_args()
    if args.verbose:
        logger.info(f"args:{args}")
        logger.info(f"unknown_args:{unknown_args}")
    assert args.work_dir, "work-dir is required"
    root = Path(args.work_dir)

    if args.log_enable:
        out_file = open(root / "out.txt", "w")
        sys.stdout = out_file
        sys.stderr = out_file
    import train

    if args.label_type == "coco":
        assert args.train_list, "train-list is required"
        assert args.val_list, "val-list is required"
        assert args.train_annotation, "train-annotation is required"
        assert args.val_annotation, "val-annatation is required"

        # 格式转换
        from converter import coco2yolo

        if not args.development:
            coco2yolo(json_path=args.train_annotation, save_path=root / "labels/train")
            coco2yolo(json_path=args.val_annotation, save_path=root / "labels/val")

        generate_coco_yaml(
            root,
            args.train_list,
            args.val_list,
            root / "labels/train",
            root / "labels/val",
            args.val_annotation,
            verbose=args.verbose,
        )

        # 导入超参数
        hyp = dict()
        if args.config:
            with open(args.config) as f:
                hyp = json.load(f)

        coco = True if args.label_type == "coco" else False
        train.run(
            unknown_args,
            data=root / "coco.yaml",
            cfg="yolov5s.yaml",
            is_coco=coco,
            project=root,
            name="result",
            device=args.device,
            verbose=args.verbose,
            **hyp,
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    main()
