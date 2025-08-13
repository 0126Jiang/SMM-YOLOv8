# -*- coding:utf-8 -*-
import argparse
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(args):
    args.batch *= 2
    model = YOLO(args.model_path)
    model.val(data=args.data_path, batch=args.batch, workers=args.workers,device=args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Training and Validation")
    parser.add_argument('--model_path', type=str, default=r'runs/detect/train3/weights/best.pt',
                        help="Path to model weights")
    parser.add_argument('--data_path', type=str, default="./data.yaml", help="Path to data YAML file")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--workers', type=int, default=8, help="Number of workers")
    parser.add_argument('--device', type=str, default="0", help="Device to use for training/validation")
    args = parser.parse_args()
    main(args)
