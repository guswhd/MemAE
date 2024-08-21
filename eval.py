import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

# CNN 모델 정의 (이전 코드와 동일해야 함)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 40 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 40 * 16)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 데이터 경로
test_dir = 'spectrogram/test'

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((160, 64)),    # 이미지 크기 조정
    transforms.ToTensor(),          # 이미지를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# 테스트 데이터셋 로드
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 로드 함수
def load_model(path='cnn_model.pth'):
    model = CNN()
    model.load_state_dict(torch.load(path))
    model.eval()  # 평가 모드로 전환
    print(f'Model loaded from {path}')
    return model

# 검증 함수 (정확도, Precision, Recall, F1-Score)
def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Accuracy 계산
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Classification report 생성 및 출력
    report = classification_report(y_true, y_pred, target_names=test_dataset.classes)
    print(report)

# 모델 로드
model = load_model('cnn_model.pth')

# 모델 평가
evaluate(model, test_loader)
