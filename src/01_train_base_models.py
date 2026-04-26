import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.utils import seed_everything

def eval_per_class(model, dataloader, device, class_names):
    #global and per-class accuracy
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
                total_correct += c[i].item()
                total_samples += 1

    overall_acc = 100. * total_correct / total_samples
    per_class_acc = {class_names[i]: 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                     for i in range(10)}
    
    return overall_acc, per_class_acc

def train_base_model(config, loaders, class_names, device):
    # this trains the model only if we don't already have the weights
    save_path = config["save_path"]
    if os.path.exists(save_path):
        print(f"Model {config['name']} already exists at {save_path}. Skipping training.")
        return

    print(f"\nTraining from scratch: {config['name']}")
    model = config["model_class"](num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    
    # scheduler
    scheduler = None
    if config["use_scheduler"]:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    epochs = config["epochs"]
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loaders['base_train'], desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
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
            
            pbar.set_postfix({'Loss': f"{loss.item():.3f}", 'Acc': f"{100.*correct/total:.1f}%"})
            
        if scheduler:
            scheduler.step()
            
    print(f"\n  Final Evaluation for {config['name']}   ")
    overall_acc, per_class_acc = eval_per_class(model, loaders['test_all'], device, class_names)
    
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    
    # save the weights and metrics
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    os.makedirs("results", exist_ok=True)
    metrics_path = f"results/{config['name'].lower()}_base_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"overall": overall_acc, "per_class": per_class_acc}, f, indent=4)
    print(f"Model and metrics saved")

def main():
    seed_everything(42)  #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Baseline Training on: {device}")

    #Nota: il forget_class=0 serve solo per preparare i dataloader - addestriamo su tutto
    loaders, class_names = get_cifar10_dataloaders(batch_size=128, forget_class=0)

    #architecture specific parameters
    configs = [
        {
            "name": "ResNet18",
            "model_class": ResNetClassifier,
            "save_path": "weights/resnet18_base_model.pth",
            "lr": 1e-3,
            "epochs": 15,
            "use_scheduler": False
        },
        {
            "name": "VGG11_bn",
            "model_class": VGG11_CIFAR10,
            "save_path": "weights/vgg11_base_model.pth",
            "lr": 3e-4,
            "epochs": 30,
            "use_scheduler": True
        },
        {
            "name": "MobileNetV2",
            "model_class": MobileNet_CIFAR10,
            "save_path": "weights/mobilenet_base_model.pth",
            "lr": 1e-3,
            "epochs": 30,
            "use_scheduler": True
        }
    ]

    for config in configs:
        train_base_model(config, loaders, class_names, device)

if __name__ == "__main__":
    main()