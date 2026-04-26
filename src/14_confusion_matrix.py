import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.surgeon import TaskArithmeticSurgeon
from unlearning.utils import seed_everything

def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_cm(y_true_base, y_pred_base, y_true_unl, y_pred_unl, classes, arch_name):
    cm_base = confusion_matrix(y_true_base, y_pred_base)
    cm_unl = confusion_matrix(y_true_unl, y_pred_unl)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Confusion Matrix: {arch_name} (Target: Ship)', fontsize=16)

    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0], cbar=False)
    axes[0].set_title('Base Model (Intact Memory)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_unl, annot=True, fmt='d', cmap='Reds', xticklabels=classes, yticklabels=classes, ax=axes[1], cbar=False)
    axes[1].set_title('Unlearned Model (Concept Destroyed)')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/cm_{arch_name.lower()}.png", dpi=300)
    plt.close()
    print(f"Matrix saved in results/plots/cm_{arch_name.lower()}.png")

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Starting Confusion Matrix Analysis on {device}")

    FORGET_CLASS = 8 
    loaders, class_names = get_cifar10_dataloaders(batch_size=256, forget_class=FORGET_CLASS)

    # best parameters found
    optimal_params = {
        "ResNet18": {"alpha": -1.0, "drop": 0.629},
        "VGG11_bn": {"alpha": -2.5, "drop": 0.990},
        "MobileNetV2": {"alpha": -1.5, "drop": 0.500}
    }

    # Architetture
    configs = [
        {"name": "ResNet18", "class": ResNetClassifier, "base": "weights/resnet18_base_model.pth", "expert": f"weights/resnet18_expert_class_{FORGET_CLASS}.pth"},
        {"name": "VGG11_bn", "class": VGG11_CIFAR10, "base": "weights/vgg11_base_model.pth", "expert": f"weights/vgg11_bn_expert_class_{FORGET_CLASS}.pth"},
        {"name": "MobileNetV2", "class": MobileNet_CIFAR10, "base": "weights/mobilenet_base_model.pth", "expert": f"weights/mobilenetv2_expert_class_{FORGET_CLASS}.pth"}
    ]

    for config in configs:
        arch_name = config['name']
        print(f"\nGenerating matrix for {arch_name}")
        
        base_model = config['class']().to(device)
        base_model.load_state_dict(torch.load(config['base'], map_location=device))
        expert_model = config['class']().to(device)
        expert_model.load_state_dict(torch.load(config['expert'], map_location=device))

        params = optimal_params[arch_name]
        unlearned_model = TaskArithmeticSurgeon.unlearn(base_model, expert_model, alpha=params['alpha'], drop_percentile=params['drop'])

        y_true_base, y_pred_base = get_predictions(base_model, loaders['test_all'], device)
        y_true_unl, y_pred_unl = get_predictions(unlearned_model, loaders['test_all'], device)

        plot_cm(y_true_base, y_pred_base, y_true_unl, y_pred_unl, class_names, arch_name)

if __name__ == "__main__":
    main()