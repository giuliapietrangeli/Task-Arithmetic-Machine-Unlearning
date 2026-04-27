import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.surgeon import TaskArithmeticSurgeon
from unlearning.utils import seed_everything

def evaluate_single_class(model, dataloader, device, target_class):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            mask = (labels == target_class)
            if mask.sum() == 0:
                continue
                
            inputs, labels = inputs[mask], labels[mask]
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100. * correct / total if total > 0 else 0

def relearn(model, train_loader, test_loader, device, target_class, max_epochs=15, target_acc=90.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-6) 
    
    history = []
    
    initial_acc = evaluate_single_class(model, test_loader, device, target_class)
    history.append(initial_acc)
    print(f"[Epoch 0] Initial accuracy: {initial_acc:.2f}%")
    
    if initial_acc >= target_acc:
        return 0, history

    for epoch in range(1, max_epochs + 1):
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            mask = (labels == target_class)
            if mask.sum() <= 1:
                continue
                
            inputs, labels = inputs[mask], labels[mask]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        acc = evaluate_single_class(model, test_loader, device, target_class)
        history.append(acc)
        print(f"[Epoch {epoch}] Accuracy: {acc:.2f}%")
        
        if acc >= target_acc:
            print(f"Target {target_acc:.2f}% reached in {epoch} epochs")
            return epoch, history
            
    print(f"Target not reached after {max_epochs} epochs")
    return max_epochs, history

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Starting Anamnesis Index evaluation on {device}")

    FORGET_CLASS = 8 # Ship
    ALPHA_MARGIN = 5.0 # Margine di tolleranza 
    MAX_EPOCHS = 30 # max training epochs
    
    loaders, class_names = get_cifar10_dataloaders(batch_size=128, forget_class=FORGET_CLASS)

    # best parameters found
    optimal_params = {
        "ResNet18": {"alpha": -1.0, "drop": 0.629},
        "VGG11_bn": {"alpha": -2.5, "drop": 0.990},
        "MobileNetV2": {"alpha": -1.5, "drop": 0.500}
    }

    configs = [
        {
            "name": "ResNet18",
            "class": ResNetClassifier,
            "base": "weights/resnet18_base_model.pth",
            "expert": f"weights/resnet18_expert_class_{FORGET_CLASS}.pth",
            "comparison": "weights/resnet18_comparison_model.pth"
        },
        {
            "name": "VGG11_bn",
            "class": VGG11_CIFAR10,
            "base": "weights/vgg11_base_model.pth",
            "expert": f"weights/vgg11_bn_expert_class_{FORGET_CLASS}.pth",
            "comparison": "weights/vgg11_comparison_model.pth"
        },
        {
            "name": "MobileNetV2",
            "class": MobileNet_CIFAR10,
            "base": "weights/mobilenet_base_model.pth",
            "expert": f"weights/mobilenetv2_expert_class_{FORGET_CLASS}.pth",
            "comparison": "weights/mobilenet_comparison_model.pth"
        }
    ]

    results = {}

    for config in configs:
        arch_name = config['name']
        print(f"\n{'='*50}\nRelearning: {arch_name}\n{'='*50}")

        if not all(os.path.exists(f) for f in [config['base'], config['expert'], config['comparison']]):
            print(f"Missing weights for {arch_name}. Skipping.")
            continue

        base_model = config['class']().to(device)
        base_model.load_state_dict(torch.load(config['base'], map_location=device))
        
        orig_acc = evaluate_single_class(base_model, loaders['test_all'], device, FORGET_CLASS)
        dynamic_target_acc = orig_acc - ALPHA_MARGIN
        print(f"Original Model Accuracy on Class {FORGET_CLASS}: {orig_acc:.2f}%")
        print(f"Dynamic Target Accuracy (Alpha={ALPHA_MARGIN}%): {dynamic_target_acc:.2f}%")

        expert_model = config['class']().to(device)
        expert_model.load_state_dict(torch.load(config['expert'], map_location=device))

        params = optimal_params[arch_name]
        unlearned_model = TaskArithmeticSurgeon.unlearn(
            base_model, expert_model, alpha=params['alpha'], drop_percentile=params['drop']
        )
        
        comparison_model = config['class']().to(device)
        comparison_model.load_state_dict(torch.load(config['comparison'], map_location=device))

        print(f"\n--- 1: Comparison Model (Native Ignorance) ---")
        epochs_comparison, hist_comp = relearn(
            comparison_model, loaders['base_train'], loaders['test_all'], 
            device, FORGET_CLASS, MAX_EPOCHS, dynamic_target_acc
        ) 

        print(f"\n--- 2: Unlearned Model (Surgical Ignorance) ---")
        epochs_unlearned, hist_unl = relearn(
            unlearned_model, loaders['base_train'], loaders['test_all'], 
            device, FORGET_CLASS, MAX_EPOCHS, dynamic_target_acc
        )

        if epochs_comparison == 0: epochs_comparison = 1 #per la divisione per zero
        ain = epochs_unlearned / epochs_comparison
        
        print(f"\n{arch_name} FINAL RESULTS:")
        print(f"Relearn Time (Comparison): {epochs_comparison} epochs")
        print(f"Relearn Time (Unlearned): {epochs_unlearned} epochs")
        print(f"Anamnesis Index (AIN): {ain:.2f}")

        results[arch_name] = {
            "Original_Accuracy": orig_acc,
            "Target_Accuracy": dynamic_target_acc,
            "Relearn_Comparison": epochs_comparison,
            "Relearn_Unlearned": epochs_unlearned,
            "AIN": ain,
            "History_Comparison": hist_comp,
            "History_Unlearned": hist_unl
        }
        
        del base_model, expert_model, unlearned_model, comparison_model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/anamnesis_index.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()