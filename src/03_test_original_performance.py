import os
import json
import torch
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.utils import seed_everything

def get_detailed_metrics(model, loaders, device, class_names, desc="Evaluating"):
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loaders['test_all'], desc=desc, leave=False):
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

    overall_acc = 100. * total_correct / total_samples if total_samples > 0 else 0
    per_class_acc = {class_names[i]: 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                     for i in range(10)}
    
    return {"overall": overall_acc, "per_class": per_class_acc}

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Baseline Evaluation on: {device}")

    loaders, class_names = get_cifar10_dataloaders(batch_size=256, forget_class=0)

    # architectures
    architectures = [
        {"name": "ResNet18", "class": ResNetClassifier, "base_weight": "weights/resnet18_base_model.pth"},
        {"name": "VGG11_bn", "class": VGG11_CIFAR10, "base_weight": "weights/vgg11_base_model.pth"},
        {"name": "MobileNetV2", "class": MobileNet_CIFAR10, "base_weight": "weights/mobilenet_base_model.pth"}
    ]

    master_results = {}

    for arch in architectures:
        arch_name = arch["name"]
        print(f"    Evaluating Architecture: {arch_name}")
        
        master_results[arch_name] = {"base_model": {}, "experts": {}}

        if os.path.exists(arch["base_weight"]):
            model = arch["class"]().to(device)
            model.load_state_dict(torch.load(arch["base_weight"], map_location=device))
            master_results[arch_name]["base_model"] = get_detailed_metrics(
                model, loaders, device, class_names, desc=f"Testing Base {arch_name}"
            )
            print(f"  Base Model Overall Acc: {master_results[arch_name]['base_model']['overall']:.2f}%")
        else:
            print(f"Base weights for {arch_name} not found.")

        print(f"Testing 10 Experts for {arch_name}")
        for i in range(10):
            expert_path = f"weights/{arch_name.lower()}_expert_class_{i}.pth"
            if os.path.exists(expert_path):
                expert_model = arch["class"]().to(device)
                expert_model.load_state_dict(torch.load(expert_path, map_location=device))
                
                metrics = get_detailed_metrics(
                    expert_model, loaders, device, class_names, desc=f"Exp {i} ({class_names[i]})"
                )
                master_results[arch_name]["experts"][f"class_{i}_{class_names[i]}"] = metrics
            else:
                print(f"  Expert for class {i} ({class_names[i]}) missing.")

    # save results
    output_dir = "results/baseline_metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "original_performance_comparison.json")
    with open(output_path, "w") as f:
        json.dump(master_results, f, indent=4)
    
if __name__ == "__main__":
    main()