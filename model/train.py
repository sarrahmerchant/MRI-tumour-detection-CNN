from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn



# typical image size is 224x224

# resize, tensor conversion, normalization

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # what do these values mean
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root='../data/Training', transform=transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 4)  # ResNet: final layer is fc (not classifier)
model = model.to(device)

# understand loss functions
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop
model.train()
for epoch in range(10):
    epoch_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    # track training intervals
    print(f'Epoch {epoch+1}/10, Avg Loss: {avg_loss:.4f}, Device: {device}')
# save next to this script so you always know where it is (not cwd-dependent)
_model_path = Path(__file__).resolve().parent / "model.pth"
torch.save(model.state_dict(), _model_path)
print(f"Saved weights to: {_model_path}")


