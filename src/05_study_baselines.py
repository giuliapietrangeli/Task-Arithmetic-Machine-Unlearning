import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.utils import seed_everything

def evaluate_unlearning(model, test_inputs, test_targets, forget_mask, retain_mask):
    # evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(test_inputs)
        _, preds = outputs.max(1)
        acc_forget = (preds[forget_mask] == test_targets[forget_mask]).float().mean().item() * 100
        acc_retain = (preds[retain_mask] == test_targets[retain_mask]).float().mean().item() * 100
    return acc_forget, acc_retain

def apply_gradient_ascent(base_model, loaders, device, epochs=3, lr=1e-4):
    #Gradient ascent
    model = copy.deepcopy(base_model).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for inputs, labels in loaders['forget_train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = -criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
    return model

def apply_random_labeling(base_model, loaders, device, epochs=3, lr=1e-4):
    #Random labeling
    model = copy.deepcopy(base_model).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for inputs, labels in loaders['forget_train']:
            inputs, labels = inputs.to(device), labels.to(device)
            random_labels = torch.randint(0, 10, labels.shape).to(device)
            random_labels = torch.where(random_labels == labels, (random_labels + 1) % 10, random_labels)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, random_labels)
            loss.backward()
            optimizer.step()
    return model

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    FORGET_CLASS = 8  # Ship
    
    loaders, class_names = get_cifar10_dataloaders(batch_size=256, forget_class=FORGET_CLASS)
    print(f"Testing Baselines on Target: {class_names[FORGET_CLASS]}")
    
    test_inputs, test_targets = [], []
    for x, y in loaders['test_all']:
        test_inputs.append(x)
        test_targets.append(y)
    test_inputs = torch.cat(test_inputs, dim=0).to(device)
    test_targets = torch.cat(test_targets, dim=0).to(device)
    
    forget_mask = (test_targets == FORGET_CLASS)
    retain_mask = (test_targets != FORGET_CLASS)

    configs = [
        {"name": "ResNet18", "class": ResNetClassifier, "base": "weights/resnet18_base_model.pth"},
        {"name": "VGG11_bn", "class": VGG11_CIFAR10, "base": "weights/vgg11_base_model.pth"},
        {"name": "MobileNetV2", "class": MobileNet_CIFAR10, "base": "weights/mobilenet_base_model.pth"}
    ]

    results = {}

    for config in configs:
        arch_name = config['name']
        print(f"RUNNING BASELINES FOR: {arch_name}")
        
        if not os.path.exists(config['base']):
            print(f"Base weights for {arch_name} missing")
            continue
            
        base_model = config['class']().to(device)
        base_model.load_state_dict(torch.load(config['base'], map_location=device))
        
        results[arch_name] = {}

        print("Applying Gradient Ascent (3 Epochs)")
        ga_model = apply_gradient_ascent(base_model, loaders, device)
        ga_forget, ga_retain = evaluate_unlearning(ga_model, test_inputs, test_targets, forget_mask, retain_mask)
        print(f"     Result -> Retain: {ga_retain:.2f}% | Forget: {ga_forget:.2f}%")
        results[arch_name]["Gradient_Ascent"] = {"retain": ga_retain, "forget": ga_forget}

        print("Applying Random Labeling (3 Epochs)")
        rl_model = apply_random_labeling(base_model, loaders, device)
        rl_forget, rl_retain = evaluate_unlearning(rl_model, test_inputs, test_targets, forget_mask, retain_mask)
        print(f"     Result -> Retain: {rl_retain:.2f}% | Forget: {rl_forget:.2f}%")
        results[arch_name]["Random_Labeling"] = {"retain": rl_retain, "forget": rl_forget}

    os.makedirs("results", exist_ok=True)
    with open("results/baselines_study.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()