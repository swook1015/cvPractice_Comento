import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def generate_heatmap_overlay():
    # 1. 특정 폴더 내의 모든 csv 파일 목록 가져오기
    # 사용자의 프로젝트 구조인 w4/output 폴더를 탐색합니다.
    csv_files = glob.glob("w4/output/*.csv") 

    if not csv_files:
        print("에러: w4/output 폴더 내에 CSV 파일이 없습니다.")
        return

    # 2. 파일 수정 시간(mtime)을 기준으로 가장 최근 파일 찾기
    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"최신 데이터 분석 중: {latest_file}")

    # 3. 데이터 불러오기
    try:
        df = pd.read_csv(latest_file)
    except Exception as e:
        print(f"파일을 읽는 중 오류 발생: {e}")
        return

    # 4. 그래프 설정 (배경 투명하게)
    # 이미지 크기는 원본 영상 비율에 맞춰 조정하는 것이 좋습니다.
    fig, ax = plt.subplots(figsize=(12, 8))

    # 점(Scatter) 그리기 
    # s=1: 점 크기, alpha=0.3: 누적된 정도를 보기 위한 투명도, c='red': 이동 궤적 색상
    ax.scatter(df['X'], df['Y'], s=2, c='red', alpha=0.3, edgecolors='none')

    # 5. 지도 위에 겹치기 위해 불필요한 요소 전부 제거
    ax.axis('off')           # 축, 수치, 테두리 제거
    fig.patch.set_alpha(0)   # 전체 배경 투명
    ax.patch.set_alpha(0)    # 그래프 영역 배경 투명

    # 6. Y축 반전 (영상 좌표계 [좌상단 0,0]와 Matplotlib 좌표계 일치시키기)
    ax.invert_yaxis()

    # 7. 투명 PNG로 저장 (transparent=True 필수)
    output_path = os.path.join("w4/output", "heatmap_overlay.png")
    plt.savefig(output_path, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
    
    plt.close() # 메모리 해제
    print(f"성공: 투명 배경 이미지가 생성되었습니다 -> {output_path}")

if __name__ == "__main__":
    generate_heatmap_overlay()