import torchvision.models as models
import torchvision.datasets as datasets
from writing_custom_datasets import CUB_200
import torchvision.transforms as transforms
import torch.utils.data
from tqdm import tqdm
import torch
import time
import os

# Data processing code
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomRotation(10),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_data = CUB_200(root='CUB-200', download=True, transform=train_transform, train=True)
test_data = CUB_200(root='CUB-200', download=True, transform=test_transform, train=False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=200, shuffle=True, num_workers=8, prefetch_factor=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=200, shuffle=True, num_workers=8, prefetch_factor=2)

# Model definition with pretrained weights
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# Replace the final fully connected layer
model.fc = torch.nn.Linear(model.fc.in_features, 200)

# Load checkpoint if exists
start_epoch = 0
best_valid_loss = float('inf')
best_accuracy = 0.0

checkpoint_path = 'model_checkpoints/best_model.pth'
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_valid_loss = checkpoint['valid_loss']
    best_accuracy = checkpoint['accuracy']
    print(f"Resumed from epoch {start_epoch} with accuracy: {best_accuracy:.2f}%")

# GPU setup
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {'params': list(model.parameters())[:-2], 'lr': 1e-5},  # Lower learning rate for pretrained layers
    {'params': list(model.parameters())[-2:], 'lr': 1e-3}  # Higher learning rate for new fc layer
])

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5, verbose=True
)


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
patience = 10
min_improvement = 0.01
patience_counter = 0

# Create directory for saving models
os.makedirs('model_checkpoints', exist_ok=True)

# Training loop
for epoch in tqdm(range(start_epoch, 10000)):
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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    # Calculate average training loss
    avg_train_loss = running_train_loss / train_batch_count

    # Validation phase
    valid_loss, accuracy = validate(model, test_loader, criterion, train_on_gpu)

    # Update learning rate scheduler
    scheduler.step(valid_loss)

    # Print statistics
    print(
        f'Epoch: {epoch + 1} \tTraining Loss: {avg_train_loss:.6f} \tValidation Loss: {valid_loss:.6f} \tAccuracy: {accuracy:.2f}%'
    )

    # Early stopping check
    if valid_loss < best_valid_loss * (1 - min_improvement):
        best_valid_loss = valid_loss
        patience_counter = 0

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_loss': valid_loss,
                'accuracy': accuracy,
            }, checkpoint_path)
            print(f'Saved new best model with accuracy: {accuracy:.2f}%')
    else:
        patience_counter += 1

    # Check if we should stop training
    if patience_counter >= patience:
        print(f'Early stopping triggered after {epoch + 1} epochs')
        break

# Load the best model after training
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']} with accuracy: {checkpoint['accuracy']:.2f}%")

