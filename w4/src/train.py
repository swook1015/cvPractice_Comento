import cv2
import numpy as np
import os
import csv
from ultralytics import YOLO
from roboflow import Roboflow
from collections import defaultdict
from datetime import datetime

if __name__ == '__main__':
    # 1. 환경 및 데이터 설정
    rf = Roboflow(api_key="yktUv9AJQjpUsW4hyUw7")
    project = rf.workspace("swook1015-ijop5").project("person-mfa1g-c3r43")
    version = project.version(1)
    
    dataset = version.download("yolov8", location="w4/data")
    now = datetime.now().strftime("%m%d_%H%M")
    dynamic_name = f"train_{now}"

    model = YOLO("yolov8n.pt") 
    print(">>> 모델 학습 시작...")
    model.train(
        data=os.path.abspath(os.path.join(dataset.location, "data.yaml")),
        epochs=30, imgsz=640, project=os.path.abspath("w4/output"),
        name=dynamic_name, exist_ok=True, lr0=0.01, batch=16, optimizer='AdamW', workers=0
    )

    