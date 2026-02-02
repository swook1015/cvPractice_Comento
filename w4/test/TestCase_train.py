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

    # 2. 모델 학습
    answer = input("모델 학습을 시작하시겠습니까? (Yes / No) :").lower().strip()
    if answer == 'Yes':
        model = YOLO("yolov8n.pt") 
        print(">>> 모델 학습 시작...")
        model.train(
            data=os.path.abspath(os.path.join(dataset.location, "data.yaml")),
            epochs=30, imgsz=640, project=os.path.abspath("w4/output"),
            name=dynamic_name, exist_ok=True, lr0=0.01, batch=16, optimizer='AdamW', workers=0
        )

    # 3. 모델 로드 (학습 직후이므로 dynamic_name 경로 그대로 사용)
    best_model_path = os.path.join("w4/output", dynamic_name, "weights", "best.pt")
    model = YOLO(best_model_path)

    # 4. 영상 분석 설정
    video_path = "w4/Test.mp4" 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"에러: {video_path} 파일을 찾을 수 없습니다."); exit()

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(f'w4/result_{dynamic_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    track_history = defaultdict(lambda: [])
    heatmap_idle = np.zeros((height, width), dtype=np.float32)

    print(">>> 영상 분석 시작: 발밑 선(빨강) / 정지(파랑 히트맵)")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                # 발밑 좌표 (Bottom Center)
                cx, cy = int(x), int(y + h / 2)
                curr_pos = (cx, cy)
                track_history[track_id].append(curr_pos)
                
                # [1. 발밑 이동 경로 그리기 (빨간 선)]
                # 최근 30프레임 동안의 경로를 선으로 연결
                points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)

                # [2. 정지 판별 및 파란색 히트맵 누적]
                # 판정 기준 완화: 최근 10프레임 동안 이동 거리가 5픽셀 미만이면 정지
                if len(track_history[track_id]) > 10:
                    prev_pos = track_history[track_id][-10]
                    dist = np.sqrt((curr_pos[0]-prev_pos[0])**2 + (curr_pos[1]-prev_pos[1])**2)
                    
                    if dist < 5: 
                        # 정지 중이면 파란색 히트맵 레이어에 강하게 표시 (반지름 키움)
                        cv2.circle(heatmap_idle, (cx, cy), 20, 1, -1)

        # 5. 히트맵 시각화 (파란색만 합성)
        idle_map = np.clip(heatmap_idle * 10, 0, 255).astype(np.uint8)
        blue_layer = np.zeros_like(frame)
        blue_layer[:, :, 0] = idle_map # B 채널

        # 원본(빨간 선 포함) + 파란 히트맵 합성
        result_frame = cv2.addWeighted(frame, 1.0, blue_layer, 0.6, 0)

        out.write(result_frame)
        cv2.imshow("Red Line: Path / Blue Heatmap: Idle", result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release(); out.release(); cv2.destroyAllWindows()

    # 6. CSV 저장
    csv_path = f'w4/tracking_{dynamic_name}.csv'
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Track_ID', 'Step_Order', 'X', 'Y'])
        for track_id, points in track_history.items():
            for i, pt in enumerate(points):
                writer.writerow([track_id, i, pt[0], pt[1]])

    print(f"완료! 영상: result_{dynamic_name}.mp4 / 데이터: tracking_{dynamic_name}.csv")