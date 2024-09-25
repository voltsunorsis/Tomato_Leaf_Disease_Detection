import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

print(f"CUDA available: {torch.cuda.is_available()}")

# Data loading function
def get_datasets(data_dir, img_size=224):
    print(f"Loading datasets from {data_dir}")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Test'), transform=transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    return train_dataset, test_dataset

# Model creation function
def create_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# Training function with k-fold cross-validation
def train_model(data_dir, num_epochs=15, batch_size=32, learning_rate=0.001, n_splits=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, test_dataset = get_datasets(data_dir)
    num_classes = len(train_dataset.classes)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        print(f'FOLD {fold+1}/{n_splits}')
        print('--------------------------------')

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler)

        model = create_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader):
                if i % 10 == 0:
                    print(f"Batch {i}/{len(train_loader)}")
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = correct / total

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = val_correct / val_total
            end_time = time.time()

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {end_time-start_time:.2f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')

        fold_results.append(best_val_acc)
        print(f"Best validation accuracy for fold {fold+1}: {best_val_acc:.4f}")

    print("\nK-FOLD CROSS VALIDATION RESULTS")
    print("--------------------------------")
    for fold, acc in enumerate(fold_results):
        print(f"Fold {fold+1}: {acc:.4f}")
    print(f"Average accuracy: {sum(fold_results)/len(fold_results):.4f}")

    return fold_results

# ... [rest of the code remains the same]

if __name__ == "__main__":
    data_dir = "H:/mini project/src/Tomato"
    fold_results = train_model(data_dir, num_epochs=5, batch_size=16)  # Reduced epochs and batch size for testing
    model_paths = [f'best_model_fold_{i}.pth' for i in range(5)]  # Assuming 5-fold CV
    evaluate_model(data_dir, model_paths)