import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv("w4/tracking_train_0201_2246.csv")

# 2. 그래프 설정 (배경 투명하게)
fig, ax = plt.subplots(figsize=(12, 8))

# 점(Scatter) 그리기 
# s=1은 점의 크기, alpha=0.3은 겹치는 곳을 보기 위한 투명도입니다.
# color='red'는 점의 색상입니다.
ax.scatter(df['X'], df['Y'], s=1, c='red', alpha=0.3, edgecolors='none')

# 3. 지도 위에 겹치기 위해 불필요한 요소 전부 제거
ax.axis('off') # 테두리, 수치, 축 제거
fig.patch.set_alpha(0) # 전체 배경 투명하게
ax.patch.set_alpha(0) # 그래프 배경 투명하게

# 4. Y축 반전 (영상 좌표계와 일치시키기)
ax.invert_yaxis()

# 5. 투명 PNG로 저장 (transparent=True가 핵심입니다)
plt.savefig('overlay_points.png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
print("투명 배경의 점 이미지 'overlay_points.png'가 생성되었습니다.")