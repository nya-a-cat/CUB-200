import torchvision.models as models
import torchvision.datasets as datasets
from writing_custom_datasets import CUB_200
import torchvision.transforms as transforms
import torch.utils.data
from tqdm import tqdm
import torch
import time
import os

# Data processing code remains the same
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomRotation(10),
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_data = CUB_200(root='CUB-200', download=True, transform=train_transform)
test_data = CUB_200(root='CUB-200', download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=200, shuffle=True, num_workers=8, prefetch_factor=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=200, shuffle=False, num_workers=8, prefetch_factor=2)

# Model definition
criterion = torch.nn.CrossEntropyLoss()
model = models.resnet50(weights=None)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
model.fc = torch.nn.Linear(model.fc.in_features, 200)

# GPU setup
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()


def validate(model, test_loader, criterion, train_on_gpu=True):
    model.eval()
    valid_loss = 0.0
    valid_total = 0
    valid_correct = 0

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            if train_on_gpu:
                batch_images = batch_images.cuda()
                batch_labels = batch_labels.cuda()

            output = model(batch_images)
            loss = criterion(output, batch_labels)
            valid_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            valid_total += batch_labels.size(0)
            valid_correct += (predicted == batch_labels).sum().item()

        avg_valid_loss = valid_loss / len(test_loader)
        valid_accuracy = 100 * valid_correct / valid_total

        return avg_valid_loss, valid_accuracy


# Early stopping parameters
patience = 10  # Number of epochs to wait for improvement
min_improvement = 0.01  # Minimum improvement required (1%)
best_valid_loss = float('inf')
patience_counter = 0
best_accuracy = 0.0

# Create directory for saving models
os.makedirs('model_checkpoints', exist_ok=True)

# Training loop
for epoch in tqdm(range(10000)):
    # Train mode
    model.train()
    running_train_loss = 0.0
    train_batch_count = 0

    for batch_images, batch_labels in tqdm(train_loader):
        if train_on_gpu:
            batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()

        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        running_train_loss += loss.item()
        train_batch_count += 1
        loss.backward()
        optimizer.step()

    # Calculate average training loss
    avg_train_loss = running_train_loss / train_batch_count

    # Validation phase
    valid_loss, accuracy = validate(model, test_loader, criterion, train_on_gpu)

    # Print statistics
    print(
        f'Epoch: {epoch + 1} \tTraining Loss: {avg_train_loss:.6f} \tValidation Loss: {valid_loss:.6f} \tAccuracy: {accuracy:.2f}%')

    # Early stopping check
    if valid_loss < best_valid_loss * (1 - min_improvement):
        # We found a better model
        best_valid_loss = valid_loss
        patience_counter = 0

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'accuracy': accuracy,
            }, f'model_checkpoints/best_model.pth')
            print(f'Saved new best model with accuracy: {accuracy:.2f}%')
    else:
        patience_counter += 1

    # Check if we should stop training
    if patience_counter >= patience:
        print(f'Early stopping triggered after {epoch + 1} epochs')
        break

# Load the best model after training
checkpoint = torch.load('model_checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']} with accuracy: {checkpoint['accuracy']:.2f}%")

