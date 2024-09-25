import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

def get_test_dataset(data_dir, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Test'), transform=transform)
    return test_dataset

def evaluate_model(data_dir, model_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = get_test_dataset(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes

    ensemble_probs = []

    for model_path in model_paths:
        model = create_model(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        all_probs = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

        ensemble_probs.append(all_probs)

    ensemble_probs = np.mean(ensemble_probs, axis=0)
    
    predicted = np.argmax(ensemble_probs, axis=1)
    true_labels = test_dataset.targets

    accuracy = np.mean(predicted == true_labels)
    print(f"Ensemble Test Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

    # Calculate ROC curve and AUC for each class
    plt.figure(figsize=(12, 10))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(np.array(true_labels) == i, ensemble_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve_ensemble.png')
    plt.close()

    print("Evaluation complete. Plots saved in 'plots' directory.")

if __name__ == "__main__":
    data_dir = "H:/mini project/src/Tomato"  # Update this path to your dataset directory
    model_paths = [f'best_model_fold_{i}.pth' for i in range(5)]  # Assuming 5-fold CV
    evaluate_model(data_dir, model_paths)