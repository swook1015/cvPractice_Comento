#### 예제 코드: 학습된 YOLOv8 모델을 사용한 객체 탐지
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import random
import glob
import os
# 학습된 YOLO 모델 로드
folder_list = glob.glob("./w3/output/train*") # 마지막으로 (가장 최근) 학습한 best.pt 찾기
if folder_list:
    latest_folder = sorted(folder_list)[-1]
    model_path = os.path.join(latest_folder, "weights", "best.pt")
    model = YOLO(model_path)

# 테스트할 이미지 불러오기
image_list = glob.glob("./w3/data/test/images/*.jpg")
image_path = random.choice(image_list)
image = cv2.imread(image_path)
# 객체 탐지 실행
results = model(image)
# 탐지된 객체 시각화
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # 좌표값 변환
        label = result.names[int(box.cls[0])] # 클래스 라벨
        confidence = box.conf[0] # 신뢰도
        # 객체 경계 상자 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# 결과 출력

#### Matplotlib을 활용한 성능 평가 시각화
# 모델 평가 결과
save_path = os.path.abspath("./w3/output")
metrics = model.val(workers=0, project = save_path)
# Precision, Recall 그래프 출력
# 1. results_dict를 통해 딕셔너리 형태로 변환하여 접근
results = metrics.results_dict

# 2. 정확한 키 이름을 사용하여 그래프 그리기
# 'metrics/precision(B)'와 'metrics/recall(B)' 키를 사용합니다.
precision = results['metrics/precision(B)']
recall = results['metrics/recall(B)']
print(f"\n[평가 결과 요약]")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall):    {recall:.4f}")
plt.bar(['Precision', 'Recall'], 
        [results['metrics/precision(B)'], results['metrics/recall(B)']], 
        color=['blue', 'green'])
plt.ylabel("Score")
plt.title("Model Evaluation Results")
plt.ylim(0, 1) # 점수는 0~1 사이이므로 범위를 고정합니다.
plt.show()
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()