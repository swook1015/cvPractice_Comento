import cv2
import numpy as np
import os
import csv
import glob
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

# 0. 결과 저장 폴더 생성
output_root = os.path.join("w4", "output")
os.makedirs(output_root, exist_ok=True)

# 1. 모델 로드 (최신 학습 폴더 자동 탐색)
folder_list = glob.glob(os.path.join(output_root, "train*"))
if not folder_list:
    print(f"에러: {output_root} 폴더 내에 학습 결과가 없습니다."); exit()

latest_folder = sorted(folder_list)[-1]
best_model_path = os.path.join(latest_folder, "weights", "best.pt")
model = YOLO(best_model_path)

# 2. 고유 파일명 설정
analysis_time = datetime.now().strftime("%m%d_%H%M%S")
output_name = f"analysis_{analysis_time}"

# 3. 영상 설정
video_path = "w4/data/video/Test.mp4"
cap = cv2.VideoCapture(video_path)
window_name = "Safety Intelligence Dashboard"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

if not cap.isOpened():
    print(f"에러: {video_path} 파일을 찾을 수 없습니다."); exit()

width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

video_output_path = os.path.join(output_root, f'result_{output_name}.mp4')
out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 데이터 및 카운팅 변수
track_history = defaultdict(lambda: [])
heatmap_idle = np.zeros((height, width), dtype=np.float32)
counted_ids = set()
total_count = 0
current_count = 0

# 필터링 설정
SMOOTHING_FACTOR = 0.4
MAX_DIST = 50 

print(f">>> 안전 모니터링 시작! 결과 저장: {output_root}")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # YOLO 추적 수행
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5)
    current_count = 0
    emergency_detected = False # 이번 프레임에 위험 객체 유무

    if results[0].boxes.id is not None:
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        current_count = len(track_ids)

        for i, track_id in enumerate(track_ids):
            # 누적 카운팅
            if track_id not in counted_ids:
                counted_ids.add(track_id)
                total_count += 1

            # --- [핵심] 누움/쓰러짐 감지 로직 ---
            x1, y1, x2, y2 = boxes_xyxy[i]
            bw = x2 - x1
            bh = y2 - y1
            aspect_ratio = bw / bh # 가로 / 세로 비율
            
            # 비율이 1.2 이상이면 누워 있는 것으로 판단 (가로가 더 김)
            is_lying = aspect_ratio > 1.2
            color = (0, 0, 255) if is_lying else (0, 255, 0)
            label = f"EMERGENCY {track_id}" if is_lying else f"Person {track_id}"
            
            if is_lying: emergency_detected = True

            # 객체 박스 표시
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 발밑 좌표 계산 (Smoothing 적용)
            x, y, w, h = boxes_xywh[i]
            cx, cy = float(x), float(y + h / 2)
            
            if len(track_history[track_id]) > 0:
                last_x, last_y = track_history[track_id][-1]
                if np.sqrt((cx - last_x)**2 + (cy - last_y)**2) < MAX_DIST:
                    cx = last_x * (1 - SMOOTHING_FACTOR) + cx * SMOOTHING_FACTOR
                    cy = last_y * (1 - SMOOTHING_FACTOR) + cy * SMOOTHING_FACTOR

            track_history[track_id].append((cx, cy))
            
            # 이동 경로 표시
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id][-20:], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)
            
            # 정지 상태 판별 및 히트맵 누적
            if len(track_history[track_id]) > 10:
                prev_pos = track_history[track_id][-10]
                if np.sqrt((cx - prev_pos[0])**2 + (cy - prev_pos[1])**2) < 5:
                    cv2.circle(heatmap_idle, (int(cx), int(cy)), 15, 1, -1)

    # 4. 실시간 현황판(Dashboard) 오버레이
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # 현황판 텍스트 정보
    status_y = 50
    header_color = (0, 0, 255) if emergency_detected else (255, 255, 255)
    header_text = "!!! EMERGENCY DETECTED !!!" if emergency_detected else "LIVE SAFETY DASHBOARD"
    
    cv2.putText(frame, header_text, (30, status_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, header_color, 2)
    cv2.putText(frame, f"| Present: {current_count}", (450, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"| Total: {total_count}", (650, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.putText(frame, f"| {datetime.now().strftime('%H:%M:%S')}", (950, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 히트맵 합성
    idle_map = np.clip(heatmap_idle * 10, 0, 255).astype(np.uint8)
    blue_layer = np.zeros_like(frame)
    blue_layer[:, :, 0] = idle_map
    result_frame = cv2.addWeighted(frame, 1.0, blue_layer, 0.6, 0)

    out.write(result_frame)
    cv2.imshow(window_name, result_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release(); out.release(); cv2.destroyAllWindows()
print(f">>> 분석 종료. 총 방문객: {total_count}명")