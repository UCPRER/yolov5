import argparse
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default=ROOT / 'workspace', help='where to store temp files')
    parser.add_argument('--device', default='', help='cpu or cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--log-path', default='', help='where to save log file')
    parser.add_argument('--dataset-type', default='coco', choices=['voc', 'coco'], help='dataset labeling method')
    parser.add_argument('--train-list', default='', help='the path of train list')


def main():
    arg = get_arg()


if __name__ == "__main__":
    main()