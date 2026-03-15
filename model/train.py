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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root='../data/Training', transform=transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    print(images.shape)
    print(labels)

model = models.resnet18(pretrained=True)
model.classifier = nn.Linear(model.fc.in_features, 4) # 4 classes

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # training loop
# for epoch in range(10):
#     for images, labels in dataloader:
#         outputs = model(images)
#         loss = loss_fn(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy.item()}')

# # save the model
# torch.save(model.state_dict(), 'model.pth')