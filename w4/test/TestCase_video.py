import cv2
import numpy as np
import os
import csv
import glob
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

# 0. 결과 저장 폴더 생성 (없으면 자동 생성)
output_root = os.path.join("w4", "output")
os.makedirs(output_root, exist_ok=True)

# 1. 최신 모델 로드
folder_list = glob.glob(os.path.join(output_root, "train*"))
if not folder_list:
    print(f"에러: {output_root} 폴더 내에 학습 결과가 없습니다."); exit()

latest_folder = sorted(folder_list)[-1]
best_model_path = os.path.join(latest_folder, "weights", "best.pt")
model = YOLO(best_model_path)

# 2. 고유 파일명 생성
analysis_time = datetime.now().strftime("%m%d_%H%M%S")
output_name = f"analysis_{analysis_time}"

# 3. 영상 설정
video_path = "w4/data/video/Test.mp4" 
cap = cv2.VideoCapture(video_path)
window_name = "Smoothed Analysis"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 창 크기 조절 가능 모드
cv2.resizeWindow(window_name, 1280, 720)       # 원하는 출력 사이즈 (가로, 세로)
if not cap.isOpened():
    print(f"에러: {video_path} 파일을 찾을 수 없습니다."); exit()

width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# [수정] 영상 저장 경로를 w4/output으로 설정
video_output_path = os.path.join(output_root, f'result_{output_name}.mp4')
out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

track_history = defaultdict(lambda: [])
heatmap_idle = np.zeros((height, width), dtype=np.float32)

# 필터링 설정 (선 튀기 방지)
SMOOTHING_FACTOR = 0.4 
MAX_DIST = 50           

print(f">>> 분석 시작! 결과 저장 위치: {output_root}")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            cx, cy = float(x), float(y + h / 2)
            
            # 이동 평균 필터 (Smoothing)
            if len(track_history[track_id]) > 0:
                last_x, last_y = track_history[track_id][-1]
                dist_check = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
                if dist_check > MAX_DIST:
                    track_history[track_id] = [] 
                
                cx = last_x * (1 - SMOOTHING_FACTOR) + cx * SMOOTHING_FACTOR
                cy = last_y * (1 - SMOOTHING_FACTOR) + cy * SMOOTHING_FACTOR

            track_history[track_id].append((cx, cy))
            
            # 발밑 이동 경로 (빨간 선)
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id][-30:], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)
            
            # 정지 판별 (파란 히트맵)
            if len(track_history[track_id]) > 10:
                prev_pos = track_history[track_id][-10]
                dist = np.sqrt((cx - prev_pos[0])**2 + (cy - prev_pos[1])**2)
                if dist < 5: 
                    cv2.circle(heatmap_idle, (int(cx), int(cy)), 20, 1, -1)

    # 히트맵 합성
    idle_map = np.clip(heatmap_idle * 10, 0, 255).astype(np.uint8)
    blue_layer = np.zeros_like(frame)
    blue_layer[:, :, 0] = idle_map 
    result_frame = cv2.addWeighted(frame, 1.0, blue_layer, 0.6, 0)

    out.write(result_frame)
    cv2.imshow("Smoothed Analysis", result_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release(); out.release(); cv2.destroyAllWindows()

# 4. CSV 데이터 저장 (w4/output으로 경로 수정)
csv_output_path = os.path.join(output_root, f'tracking_{output_name}.csv')
with open(csv_output_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Track_ID', 'Step_Order', 'X', 'Y'])
    for track_id, points in track_history.items():
        for i, pt in enumerate(points):
            writer.writerow([track_id, i, int(pt[0]), int(pt[1])])

print("-" * 30)
print(f"모든 파일 저장 완료: {output_root}")
print(f"영상: result_{output_name}.mp4")
print(f"데이터: tracking_{output_name}.csv")
print("-" * 30)