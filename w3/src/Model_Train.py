from ultralytics import YOLO
from roboflow import Roboflow
import os
from datetime import datetime

if __name__ == '__main__':
    # 1. Roboflow 설정
    rf = Roboflow(api_key="yktUv9AJQjpUsW4hyUw7")
    project_rf = rf.workspace("swook1015-ijop5").project("person-car-dog-bicycle-s7n9b")
    version = project_rf.version(1)
    
    # 데이터 위치를 w3/data로 지정
    dataset = version.download("yolov8", location="w3/data")
    
    # 학습 폴더 이름을 위한 시간 지정
    now = datetime.now().strftime("%m%d_%H%M")
    dynamic_name = f"train_{now}" 

    # 2. 모델 로드
    model = YOLO("yolov8n.pt") 

    # 3. 학습 시작
    model.train(
    # 1. 경로 및 기본 설정
    data=os.path.abspath(f"{dataset.location}/data.yaml"), # 학습에 필요한 데이터셋 정보(이미지 경로, 클래스 이름 등)가 담긴 파일 경로입니다.
    epochs=30,                                            # 전체 데이터셋을 총 30번 반복해서 학습합니다. 숫자가 클수록 오래 공부합니다.
    imgsz=640,                                            # 입력 이미지의 크기를 640x640 픽셀로 맞추어 학습합니다.
    project=os.path.abspath("w3/output"),                 # 학습 결과(모델, 그래프 등)를 저장할 최상위 폴더 경로입니다.
    name=dynamic_name,                                    # 학습 시마다 고유한 폴더 이름(예: train_0126_0128)을 만들어 결과물을 구분합니다.
    exist_ok=True,                                        # 동일한 이름의 폴더가 있어도 에러를 내지 않고 덮어쓰거나 이어서 작업합니다.

    # 2. 하이퍼파라미터 (공부 방법 설정)
    lr0=0.01,                                             # 초기 학습률입니다. 모델이 정답을 찾아가는 '발걸음의 크기'를 결정합니다.
    batch=16,                                             # 배치 사이즈입니다. 그래픽카드가 한 번에 처리할 이미지 장수입니다.
    optimizer='AdamW',                                    # 최적화 알고리즘입니다. 오차를 줄여나가는 수학적 방법론으로, 최근 가장 많이 쓰이는 방식입니다.

    # 3. 데이터 증강 (실전 대비 랜덤 훈련)
    degrees=15.0,                                         # 이미지를 ±15도 범위 내에서 랜덤하게 회전시켜 각도 변화에 강하게 만듭니다.
    hsv_v=0.4,                                            # 이미지의 밝기를 ±40% 범위 내에서 랜덤하게 조절합니다.
    fliplr=0.5,                                           # 50% 확률로 이미지를 좌우 반전시켜 학습합니다.
    mosaic=1.0,                                           # 4장의 이미지를 하나로 합쳐 모델이 물체의 일부만 보고도 맞히게 훈련합니다.

    # 4. 시스템 설정
    workers=0                                             # 데이터 로딩에 사용할 프로세스 수입니다. 윈도우 환경에서는 충돌 방지를 위해 0으로 설정합니다.
)