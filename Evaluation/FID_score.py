import numpy as np
from imageio import imread
from numpy import cov, trace
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import os

# Function to calculate FID between two sets of activations
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Define a custom dataset class for images
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = imread(img_path)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image

# Initialize the feature extractor model (InceptionV3)
def initialize_model():
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

# Function to extract activations from a dataset
def extract_activations(model, dataloader):
    activations = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            activations.append(outputs.cpu().numpy())

    activations = np.concatenate(activations, axis=0)
    return activations

if __name__ == "__main__":
    # 데이터셋 경로 정의
    folder1 = "C:/Users/Lab/PycharmProjects/pythonProject/cycleGAN/output/B"
    folder2 = "C:/Users/Lab/PycharmProjects/pythonProject/datasets/spine(axial)/train/B"

    # Define transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet mean and std
    ])

    # Create datasets
    dataset1 = ImageDataset(root=folder1, transform=transform)
    dataset2 = ImageDataset(root=folder2, transform=transform)

    # Create dataloaders for activations extraction
    dataloader1 = DataLoader(dataset1, batch_size=64, shuffle=False)
    dataloader2 = DataLoader(dataset2, batch_size=64, shuffle=False)

    # Initialize the feature extractor model
    model = initialize_model()

    # Extract activations from the datasets
    activations1 = extract_activations(model, dataloader1)
    activations2 = extract_activations(model, dataloader2)

    # Calculate FID
    fid_score = calculate_fid(activations1, activations2)
    print('FID Score:', fid_score)

