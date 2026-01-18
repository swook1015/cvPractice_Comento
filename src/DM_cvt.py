import cv2
import numpy as np
import os
import glob

if not os.path.exists('output'):
    print("메인 폴더에 output파일이 없으므로 생성")
    os.makedirs('output')

ims = glob.glob('data/*.jpg') # data 폴더 안에 있는 파일 전체를 ims에 할당 (추가)

if ims:
    print(f"{len(ims)}개의 이미지를 감지") # 사용자에게 프로그램이 몇 개의 이미지를 감지했는지 보여줌 (추가)
    for i in ims:
        image = cv2.imread(i)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        base_name = os.path.basename(i)
        save_path = f'output/cvt_{base_name}'

        cv2.imwrite(save_path, depth_map) # 결과 이미지 저장 (추가)
        cv2.imshow(f'Original Image_{base_name}', image)
        cv2.imshow(f'Depth Map_{base_name}', depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("이미지를 감지 못함")