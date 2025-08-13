# -*- coding:utf-8 -*-
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    yaml_path = 'yolov8n-mca-slimneck.yaml'
    model = YOLO(yaml_path)
    print(model.model)


if __name__ == '__main__':
    main()
