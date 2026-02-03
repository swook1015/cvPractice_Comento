# 🔍 Intelligent Flow & Safety Analysis System (YOLOv8 + IoT)

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Mobius](https://img.shields.io/badge/oneM2M-Mobius-blue?style=for-the-badge)](https://github.com/IoTKETI/Mobius)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)](https://opencv.org/)

</div>

## 📝 프로젝트 소개
본 프로젝트는 YOLOv8을 활용하여 객체의 실시간 이동 경로(Trajectory)와 체류 지점(Dwell-time)을 분석하고, 나아가 낙상 및 쓰러짐 사고를 실시간으로 감지하는 지능형 통합 모니터링 시스템입니다.

단순 탐지를 넘어 객체별 고유 ID를 부여하고 동선 데이터를 수집하며, 분석된 데이터는 표준 IoT 미들웨어인 Mobius(oneM2M)와 연동하여 스마트 관제 플랫폼으로 확장하는 것을 목표로 합니다.

### 🚀 핵심 차별점 및 기능
* **지능형 낙상 감지 (Safety):** 객체 박스의 가로/세로 비율($Aspect\ Ratio > 1.2$) 분석을 통해 쓰러짐 상황을 자동 감지하고 즉각적인 경고 메시지를 출력합니다.
* **실시간 대시보드 (Dashboard):** 현재 화면 인원(Present)과 누적 방문객(Total)을 실시간으로 집계하여 상단 오버레이로 표시합니다.
* **동적 시각화:** 실시간 발밑 경로(Red Line) 및 정지 구간 히트맵(Blue Heatmap) 중첩 기술을 적용하여 유동 인구의 특성을 한눈에 파악합니다.
* **데이터 정밀화:** 이동 평균 필터(Smoothing Filter)를 적용하여 좌표 떨림 및 튀는 현상을 최소화했습니다.

---

## ⚙️ 주요 알고리즘 상세

| **낙상 감지 알고리즘 (Safety)** | **동선 및 히트맵 분석 (Flow)** |
| :---: | :---: |
| <img width="350" height="350" alt="스크린샷 2026-02-03 185940" src="https://github.com/user-attachments/assets/34a98cb5-1cba-4b4b-9910-fc724d04382e" /> | <img width="350" height="350" alt="스크린샷 2026-02-03 185933" src="https://github.com/user-attachments/assets/22d0b1fe-21dd-4aee-bb77-2aa6afe1188d" />
 |
| **비율 기반 분석 ($AR > 1.2$)** | **실시간 경로 추적 및 시각화** |
| 객체의 가로/세로 비율 변화를 실시간으로 <br> 계산하여 낙상 사고를 즉각 감지합니다. | ByteTrack과 이동 평균 필터를 결합하여 <br> 정밀한 동선 추적 및 히트맵을 생성합니다. |


### 1. 쓰러짐 감지 로직 (Fall Detection)
사람 객체가 쓰러졌을 경우, 바운딩 박스의 형태가 세로형에서 가로형으로 변하는 특성을 이용합니다.
$$Aspect\ Ratio = \frac{Width}{Height}$$
* **정상 (Normal):** $Aspect\ Ratio < 1.0$ (세로가 더 길음)
* **위험 (Emergency):** $Aspect\ Ratio > 1.2$ (가로가 더 길어짐) $\rightarrow$ **Red Box & Alert** 표시

### 2. 실시간 카운팅 (Dynamic Counting)
`set()` 자료구조를 활용하여 고유 Track ID를 관리함으로써, 화면에서 사라졌다 다시 나타나도 동일 인물은 누적 카운트에 중복 집계되지 않도록 설계되었습니다.

---

## 🛠 기술 스택 및 환경
| 구분 | 기술 스택 | 비고 |
| :---: | :--- | :--- |
| **AI Framework** | YOLOv8 (Ultralytics) | Object Detection & Tracking |
| **Tracking** | ByteTrack | Multi-object ID Tracking |
| **Vision Library** | OpenCV (Python) | UI Dashboard & Image Processing |
| **Middleware** | Mobius (oneM2M) | IoT Data Management ( 예정 ) |
| **Data Analysis** | Pandas / CSV | Time-series Coordinate Data |

---

## 📂 폴더 구조 (w4)
* **w4/data**: 학습 데이터셋
* **w4/data_video**: 분석 대상 영상(`Input.mp4`)
* **w4/output**: 학습 모델(`best.pt`), 결과 영상(`.mp4`), 트래킹 로그(`.csv`)
* **w4/src**: 
    * `video2tracking.py`: 메인 분석 및 실시간 대시보드 실행 코드
    * `train.py`: YOLOv8 커스텀 데이터 학습 스크립트
* **w4/test**: 단위 기능 테스트 코드

---

## 🏃 실행 방법
```bash
# 1. 모델 학습 (필요 시)
python w4/src/train.py

# 2. 지능형 대시보드 및 분석 실행
python w4/src/video2tracking.py
