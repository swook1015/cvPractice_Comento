import cv2
import numpy as np
from datasets import load_dataset
import os
import random

# 1. 저장 폴더 생성
os.makedirs('preprocessed_samples', exist_ok=True)

# 2. 데이터셋 로드
ds = load_dataset("ethz/food101", split="train", streaming=True)
it = iter(ds)

for i in range(1, 6):
    sample = next(it)
    # PIL 이미지를 OpenCV BGR 형식으로 변환
    image = cv2.cvtColor(np.array(sample['image']), cv2.COLOR_RGB2BGR)
    
    # [전처리 1] 크기 조정 (224x224)
    resized = cv2.resize(image, (224, 224))

    # [전처리 2]
    if random.random() > 0.5:
        resized = cv2.flip(resized, 1)
    angle = random.uniform(-15, 15)
    matrix = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
    resized = cv2.warpAffine(resized, matrix, (224, 224))
    
    # [전처리 3] 색상 변환
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # [전처리 4] 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
     
    # [전처리 5] 정규화
    normalized = blurred / 255.0
    
    # 이미지 저장
    final_image = (normalized * 255).astype(np.uint8)
    cv2.imwrite(f'preprocessed_samples/sample_{i}.png', final_image)
    print(f"[{i}/5] 전처리 및 증강 완료")

    