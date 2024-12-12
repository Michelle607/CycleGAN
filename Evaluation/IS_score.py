import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image

# 경로에 있는 모든 이미지 파일을 로드합니다.
def load_images_from_folder(folder, transform):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)
    return torch.stack(images)

# Inception Score 계산 함수
def calculate_inception_score(p_yx, eps=1E-16):
    # p(y) 계산
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # 각 이미지에 대한 KL divergence 계산
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # 클래스에 대해 합산
    sum_kl_d = kl_d.sum(axis=1)
    # 이미지에 대해 평균
    avg_kl_d = np.mean(sum_kl_d)
    # 로그를 풀어줌
    is_score = np.exp(avg_kl_d)
    return is_score

# Inception 네트워크 준비
model = inception_v3(pretrained=True, transform_input=False)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 두 경로에서 이미지를 로드
folder1 = "C:/Users/Lab/PycharmProjects/pythonProject/cycleGAN/output/A"
folder2 = "C:/Users/Lab/PycharmProjects/pythonProject/datasets/spine(axial)/train/A"
images1 = load_images_from_folder(folder1, transform)
images2 = load_images_from_folder(folder2, transform)

# 모든 이미지를 하나의 텐서로 결합
images = torch.cat((images1, images2), 0)

# Inception 네트워크를 사용하여 클래스 확률 예측
with torch.no_grad():
    outputs = model(images)
    p_yx = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

# Inception Score 계산
score = calculate_inception_score(p_yx)
print(f'Inception Score: {score}')
