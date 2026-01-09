import cv2
import numpy as np
from datasets import load_dataset
import os

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
    
    # [전처리 2] 색상 변환 (Grayscale)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # [전처리 3] 노이즈 제거 (Gaussian Blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # [전처리 4] Normalize (0~255를 0~1 사이로 정규화)
    normalized = blurred / 255.0
    
    # 이미지 저장 (정규화된 데이터는 시각화를 위해 다시 255를 곱해 저장)
    final_image = (normalized * 255).astype(np.uint8)
    cv2.imwrite(f'preprocessed_samples/sample_{i}.png', final_image)
    print(f"[{i}/5] 전처리 완료")