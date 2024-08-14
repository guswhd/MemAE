import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_and_resize_image(img_path, size=(160, 64)):
    # 이미지 로드
    image = Image.open(img_path)
    # 이미지 리사이즈
    image = image.resize(size)
    # numpy 배열로 변환
    image = np.array(image)
    return image

def visualize_images(image_dir, size=(160, 64), num_images=5):
    # 디렉토리에서 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    # 지정된 수만큼의 이미지를 로드 및 시각화
    for i, img_file in enumerate(image_files[:num_images]):
        img_path = os.path.join(image_dir, img_file)
        image = load_and_resize_image(img_path, size=size)
        
        # 이미지 시각화
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Image {i + 1}")
        plt.axis('off')
    
    # 전체 이미지 시각화
    plt.show()

# 예시: './spectrogram/Male' 디렉토리에서 이미지 시각화
visualize_images('./spectrogram/Male', size=(160, 64), num_images=1)
