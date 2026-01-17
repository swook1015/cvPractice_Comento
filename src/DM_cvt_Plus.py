#### 심화 코드: Depth Map을 기반으로 3D 포인트 클라우드 생성
import cv2
import numpy as np
import os
import glob

if not os.path.exists('output'):
    print('메인폴더에 output 파일이 없으므로 생성')
    os.makedirs('output')

ims = glob.glob('data/*.jpg')

if ims:
    print(f'{len(ims)} 개의 이미지를 감지했습니다.')
    for i in ims:
        image = cv2.imread(i)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        depth_map =  cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        # 3D 포인트 클라우드 변환
        h, w = depth_map.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        Z = gray.astype(np.float32) # Depth 값을 Z 축으로 사용
        # 3D 좌표 생성
        points_3d = np.dstack((X, Y, Z))
        # 결과 이미지 저장 (추가)
        cv2.imwrite('../output/result2.jpg', depth_map)
        # 결과 출력
        cv2.imshow('Depth Map', depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('이미지가 없습니다.')