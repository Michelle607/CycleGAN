import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# 커스텀 이미지 데이터셋 클래스
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path  # 이미지 경로도 반환

# PSNR 점수 계산 함수
def calculate_psnr(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    return psnr(img1, img2)

if __name__ == "__main__":
    # 데이터셋 경로 정의
    folder1 = "C:/Users/Lab/PycharmProjects/pythonProject/cycleGAN/output(6)/A"
    folder2 = "C:/Users/Lab/PycharmProjects/pythonProject/datasets/spine(axial)/train/A"

    # 이미지 전처리 변환 정의
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # 데이터셋 생성
    dataset1 = ImageDataset(root=folder1, transform=transform)
    dataset2 = ImageDataset(root=folder2, transform=transform)

    # 데이터로더 생성
    dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False)
    dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=False)

    # PSNR 점수 계산
    psnr_scores = []

    for (img1, path1), (img2, path2) in zip(dataloader1, dataloader2):
        img1 = img1.squeeze().permute(1, 2, 0).numpy() * 255
        img2 = img2.squeeze().permute(1, 2, 0).numpy() * 255

        img1 = Image.fromarray(img1.astype('uint8'))
        img2 = Image.fromarray(img2.astype('uint8'))

        score = calculate_psnr(img1, img2)
        psnr_scores.append(score)

    avg_psnr = np.mean(psnr_scores)
    print('Average PSNR Score:', avg_psnr)
