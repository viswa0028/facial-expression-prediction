import cv2 as cv
import torch
from torchvision import transforms
import torch.nn
from PIL import Image
class Convolution(torch.nn.Module):
    def __init__(self, num_classes):
        super(Convolution, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.poolinglayer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(18432, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.poolinglayer(self.relu(self.conv2(x)))
        x = self.poolinglayer(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Convolution(num_classes=7)
model.load_state_dict(torch.load('facial-expression-prediction/facial_expression_cnn.pth'))
model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
]
)
image = Image.open('YOUR IMAGE PATH').convert('RGB')
image = transform(image).unsqueeze(0).to(device)  

# Make prediction
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
predicted_label = class_labels[predicted_class]

print(f"Predicted Expression: {predicted_label}")
