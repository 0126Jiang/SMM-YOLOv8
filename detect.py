# -*- coding:utf-8 -*-
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():

    model_path = 'yolov8n.pt'

    model = YOLO(model_path)
    model.predict(model=model_path, source='', save=True)


if __name__ == '__main__':
    main()
