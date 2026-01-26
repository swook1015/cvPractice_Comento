#### 예제 코드: YOLOv8을 활용한 객체 탐지 모델 학습
from ultralytics import YOLO
import os
# YOLOv8 모델 로드
model = YOLO("yolov8n.pt") # YOLOv8 기본 모델 사용
# 사용자 데이터셋으로 학습 (data.yaml 파일 필요)
from roboflow import Roboflow # YOLO 학습,시험데이터에 적합한 RoboFlow를 이용함
rf = Roboflow(api_key="yktUv9AJQjpUsW4hyUw7")
project = rf.workspace("swook1015-ijop5").project("face-pflhs-vmob7")
version = project.version(1)
dataset = version.download("yolov8", location = "./w3/test/test_data")
model.train(data="w3/test/test_data/data.yaml",
             epochs=30, 
             imgsz=640, 
             project = os.path.abspath("w3/test/test_output") ,
             workers = 0
             )

# RoboFlow에서 YOLO8을 위한 데이터를 받고, 
# 그 데이터들을 이용하여 학습 실행. 현재 데이터셋은 데이터 수가 적어서 에폭을 30으로 임시 지정함.