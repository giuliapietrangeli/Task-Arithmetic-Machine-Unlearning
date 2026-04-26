import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.utils import seed_everything

def eval_comparison_model(model, dataloader, device, forget_class):
    # evaluate
    model.eval()
    correct_retain = 0
    total_retain = 0
    correct_forget = 0
    total_forget = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            retain_mask = (labels != forget_class)
            if retain_mask.sum() > 0:
                total_retain += retain_mask.sum().item()
                correct_retain += predicted[retain_mask].eq(labels[retain_mask]).sum().item()
                
            forget_mask = (labels == forget_class)
            if forget_mask.sum() > 0:
                total_forget += forget_mask.sum().item()
                correct_forget += predicted[forget_mask].eq(labels[forget_mask]).sum().item()

    acc_retain = 100. * correct_retain / total_retain if total_retain > 0 else 0
    acc_forget = 100. * correct_forget / total_forget if total_forget > 0 else 0
    
    return acc_retain, acc_forget

def train_comparison_model(config, loaders, device, forget_class):
    # train model without forget class
    save_path = config["save_path"]
    if os.path.exists(save_path):
        print(f"Comparison Model {config['name']} already exists. Skipping.")
        return

    print(f"Training Comparison Model (from scratch, no {forget_class}): {config['name']}")
    
    model = config["model_class"](num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    
    scheduler = None
    if config["use_scheduler"]:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    epochs = config["epochs"]
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(loaders['base_train'], desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            mask = (labels != forget_class)
            
            if mask.sum() == 0:
                continue
                
            inputs = inputs[mask]
            labels = labels[mask]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.3f}"})
            
        if scheduler:
            scheduler.step()
            
    print(f"Final Evaluation for {config['name']} Comparison Model")
    acc_retain, acc_forget = eval_comparison_model(model, loaders['test_all'], device, forget_class)
    
    print(f"Retain Accuracy (9 classes): {acc_retain:.2f}%")
    print(f"Forget Accuracy (Ship): {acc_forget:.2f}%")
    
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), save_path)

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    FORGET_CLASS = 8 # Ship
    loaders, class_names = get_cifar10_dataloaders(batch_size=128, forget_class=FORGET_CLASS)
    print(f"Starting Comparison Training on: {device}")
    print(f"Target class excluded from training: {class_names[FORGET_CLASS]}")

    # architetture
    configs = [
        {
            "name": "ResNet18",
            "model_class": ResNetClassifier,
            "save_path": "weights/resnet18_comparison_model.pth",
            "lr": 1e-3,
            "epochs": 15,
            "use_scheduler": False
        },
        {
            "name": "VGG11_bn",
            "model_class": VGG11_CIFAR10,
            "save_path": "weights/vgg11_comparison_model.pth",
            "lr": 3e-4,
            "epochs": 30,
            "use_scheduler": True
        },
        {
            "name": "MobileNetV2",
            "model_class": MobileNet_CIFAR10,
            "save_path": "weights/mobilenet_comparison_model.pth",
            "lr": 1e-3,
            "epochs": 30,
            "use_scheduler": True
        }
    ]

    for config in configs:
        train_comparison_model(config, loaders, device, FORGET_CLASS)

if __name__ == "__main__":
    main()