# 21316651 Adesayo Akintunde
print("STARTING SCRIPT...") # Troubleshooting line to check if the script is running correctly. If this line is printed, it means the script has started executing without any immediate syntax errors.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), #Convert data set to Tensor for PyTorch
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'test': transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
}

# Correct dataset path
data_dir = r"C:\Users\adyak\Documents\DL_Project\Emotions Dataset"

#Create data Loaders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train','test']}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=0)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','test']} #Checking the number of records in train and in testing
print(dataset_sizes)

class_names = image_datasets['train'].classes #storing class names in a variable
print(class_names)

# Load the pre-trained ResNet-18 model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

# Unfreeze layer3, layer4 and final layer
for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

#Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss() #Cross-entropy loss for multi-class classification
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

#Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

train_losses, test_losses = [], []
train_accs, test_accs = [], []

best_acc = 0.0
best_model_path = 'best_emotion_recognition_resnet18.pth'

# Training the model
num_epochs = 12
for epoch in range(num_epochs):
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        current_acc = epoch_acc.item()

        print(f'Epoch {epoch}/{num_epochs - 1} - {phase} Loss: {epoch_loss:.4f} Acc: {current_acc:.4f}')

        if phase == 'train':
            train_losses.append(epoch_loss)
            train_accs.append(current_acc)
        else:
            test_losses.append(epoch_loss)
            test_accs.append(current_acc)

            if current_acc > best_acc:
                best_acc = current_acc
                torch.save(model.state_dict(), best_model_path)

    scheduler.step()

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig('loss_curve.png')
plt.show()

plt.figure()
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig('accuracy_curve.png')
plt.show()

print("Best test accuracy:", best_acc)
print("SCRIPT COMPLETED.")

# Load the best saved model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)
model.eval()

# Load and preprocess a new image for prediction
image_path = r"C:\Users\adyak\Documents\DL_Project\testimage.jpg" # Update with the path to your test image
image = Image.open(image_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Perform inference and get predicted probabilities
with torch.no_grad():
    output = model(input_batch.to(device))
    probs = torch.softmax(output, dim=1)

_, predicted = torch.max(probs, 1)
predicted_label = class_names[predicted.item()]

print(f'Predicted emotion: {predicted_label}')
print("Class probabilities:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {probs[0][i].item():.4f}")

#Show the image and the predicted label
image = np.array(image)
plt.imshow(image)
plt.axis('off')
plt.text(10, 30, f'Predicted: {predicted_label}', color='red', fontsize=12)
plt.show()