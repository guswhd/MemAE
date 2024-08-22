import os
import shutil
from sklearn.model_selection import train_test_split

# base_dir 입력 받기
base_dir = input("abnormal, normal 디렉토리의 부모 디렉토리 절대 경로: ")

# 원본 디렉토리 경로 설정
original_abnormal_dir = os.path.join(base_dir, 'abnormal')
original_normal_dir = os.path.join(base_dir, 'normal')

# 타겟 디렉토리 경로 설정
train_abnormal_dir = os.path.join(base_dir, 'train', 'abnormal')
train_normal_dir = os.path.join(base_dir, 'train', 'normal')
test_abnormal_dir = os.path.join(base_dir, 'test', 'abnormal')
test_normal_dir = os.path.join(base_dir, 'test', 'normal')

# 디렉토리 생성
os.makedirs(train_abnormal_dir, exist_ok=True)
os.makedirs(train_normal_dir, exist_ok=True)
os.makedirs(test_abnormal_dir, exist_ok=True)
os.makedirs(test_normal_dir, exist_ok=True)

# 이미지 파일 목록 가져오기
abnormal_images = os.listdir(original_abnormal_dir)
normal_images = os.listdir(original_normal_dir)

# 데이터 분할 (8:2 비율)
train_abnormal, test_abnormal = train_test_split(abnormal_images, test_size=0.2, random_state=42)
train_normal, test_normal = train_test_split(normal_images, test_size=0.2, random_state=42)

# 파일 이동 함수 정의
def move_files(file_list, src_dir, dst_dir):
    for file_name in file_list:
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.move(src_path, dst_path)

# 파일 이동
move_files(train_abnormal, original_abnormal_dir, train_abnormal_dir)
move_files(test_abnormal, original_abnormal_dir, test_abnormal_dir)
move_files(train_normal, original_normal_dir, train_normal_dir)
move_files(test_normal, original_normal_dir, test_normal_dir)

print("데이터 분할 및 이동이 완료되었습니다.")
