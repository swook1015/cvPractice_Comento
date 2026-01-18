#### 심화 코드: Depth Map을 기반으로 3D 포인트 클라우드 생성
import cv2
import numpy as np
import os
import glob

if not os.path.exists('output'): # 해당 output 파일폴더가 없으면 생성
    print('메인폴더에 output 파일이 없으므로 생성')
    os.makedirs('output')

ims = glob.glob('data/*.jpg') # data 파일 안에 있는 .jpg를 ims에 할당

if ims: #프로그램이 data 파일 폴더 안에 있는 .jpg를 감지했으면 실행
    print(f'{len(ims)} 개의 이미지를 감지했습니다.')# 사용자에게 프로그램이 몇 개의 이미지를 감지했는지 보여줌 (추가)
    for i in ims:# 모든 파일에 적용하기 위해 for문 실행
        image = cv2.imread(i)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        depth_map =  cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        base_name = os.path.basename(i) #해당 파일 이름 저장
        save_path = f'output/{base_name}'#해당 파일 경로 저장

        # 3D 포인트 클라우드 변환
        h, w = depth_map.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        Z = gray.astype(np.float32) # Depth 값을 Z 축으로 사용
        # 3D 좌표 생성
        points_3d = np.dstack((X, Y, Z))

        save_point = f'output/{base_name}_3d_Points.npy' #3D포인트를 저장하기 위한 경로 저장
        np.save(save_point, points_3d) #3D포인트 저장

        # 결과 이미지 저장 (추가)
        cv2.imwrite(save_path, depth_map)
        # 결과 출력 (추가)
        cv2.imshow(save_path, depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('이미지가 없습니다.')