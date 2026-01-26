

# 🔍 Computer Vision: Object Detection Project


[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8%20%2F%20v11-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)

</div>

<br>

## 📝 프로젝트 소개
이 프로젝트는 **Roboflow 데이터셋**을 활용하여 YOLO 기반의 객체 탐지 모델을 구축하고 실습하는 과정입니다. <br>현재 **Week 3 (w3)** 단계인 모델링 및 추론 최적화를 진행하고 있습니다.

<br>

## 🛠 기술 스택 및 환경
프로젝트 수행을 위해 아래 라이브러리들을 활용하였습니다.

| 라이브러리 | 용도 | 비고 |
| :--- | :--- | :--- |
| **PyTorch** | 딥러닝 모델 학습 및 연산 | Core Engine |
| **Ultralytics** | YOLO 모델 학습 및 추론 인터페이스 | Framework |
| **OpenCV** | 이미지 전처리 및 결과 시각화 | Computer Vision |
| **Matplotlib** | 학습 결과(Loss, mAP) 시각화 | Analysis |

<br>

## 📂 폴더 구조
주요 작업 내용은 `w3` 폴더에 집중되어 있습니다.

* **w3/src**: 학습 및 추론을 위한 핵심 소스 코드
* **w3/data**: `.gitkeep`을 통해 데이터셋 저장 구조 유지 (train/valid/test)
* **w3/output**: 학습된 모델 가중치(`.pt`) 및 성능 리포트 저장
* **w3/test**: Face 데이터셋을 이용한 Test Case
* **data.yaml**: 데이터셋 경로 및 클래스 설정 파일

<br>

## ⚙️ 실행 방법
```bash
# 1. 필수 라이브러리 설치
pip install torch torchvision opencv-python matplotlib ultralytics

# 2. 학습 실행
python w3/src/Model_Train.py

# 3. 객체 감지 실행
python w3/src/YOLOv8_Object_Direction.py
```

<div align="center">
  <table>
    <tr>
      <td><img width="858" height="733" alt="스크린샷 2026-01-26 011127" src="https://github.com/user-attachments/assets/2d92a754-aea7-48e9-9cfe-afd07ac54d98" /></td>
      <td><img width="640" height="534" alt="스크린샷 2026-01-25 194843" src="https://github.com/user-attachments/assets/e02b2b45-8f14-4b08-a276-5f9bc77fe1ad" /></td>
      <td><img width="922" height="918" alt="스크린샷 2026-01-26 012223" src="https://github.com/user-attachments/assets/5f734525-bcbe-47f8-b689-de4f45ead1a9" /><div align="center"></td>
    </tr>
    <tr align="center">
      <td>w3/data에 저장된 RoboFlow 데이터셋</td>
      <td>Object_Direction된 모습</td>
      <td>데이터 증강이 적용된 모습</td>
    </tr>
  </table>
</div>

<br>

## Author
허성욱(swook1015@gwnu.ac.kr)
