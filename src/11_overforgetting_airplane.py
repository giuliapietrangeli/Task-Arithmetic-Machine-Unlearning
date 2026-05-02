import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.surgeon import TaskArithmeticSurgeon
from unlearning.utils import seed_everything

def evaluate_overforgetting(unlearned_model, dataloader, device, target_class, neighbor_class):
    unlearned_model.eval() #evaluation
    
    correct_neighbor, total_neighbor = 0, 0
    correct_others, total_others = 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = unlearned_model(inputs)
            _, predicted = outputs.max(1)
            
            mask_neighbor = (labels == neighbor_class)
            if mask_neighbor.sum() > 0:
                total_neighbor += mask_neighbor.sum().item()
                correct_neighbor += predicted[mask_neighbor].eq(labels[mask_neighbor]).sum().item()
                
            mask_others = (labels != target_class) & (labels != neighbor_class)
            if mask_others.sum() > 0:
                total_others += mask_others.sum().item()
                correct_others += predicted[mask_others].eq(labels[mask_others]).sum().item()
                
    acc_neighbor = 100. * correct_neighbor / total_neighbor if total_neighbor > 0 else 0 #accuracy on neighbor class
    acc_others = 100. * correct_others / total_others if total_others > 0 else 0 #accuracy on other classes
    
    return acc_neighbor, acc_others

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Starting Over-forgetting Analysis on {device}")

    FORGET_CLASS = 8 # Ship
    NEIGHBOR_CLASS = 0 # Airplane
    
    loaders, class_names = get_cifar10_dataloaders(batch_size=256, forget_class=FORGET_CLASS)
    print(f"Target Forget Class: {class_names[FORGET_CLASS]}")
    print(f"Neighbor Class (At Risk): {class_names[NEIGHBOR_CLASS]}")

    # best params
    optimal_params = {
        "ResNet18": {"alpha": -1.0, "drop": 0.629},
        "VGG11_bn": {"alpha": -2.5, "drop": 0.990},
        "MobileNetV2": {"alpha": -1.5, "drop": 0.500}
    }

    #configs of architectures
    configs = [
        {
            "name": "ResNet18",
            "class": ResNetClassifier,
            "base": "weights/resnet18_base_model.pth",
            "expert": f"weights/resnet18_expert_class_{FORGET_CLASS}.pth",
        },
        {
            "name": "VGG11_bn",
            "class": VGG11_CIFAR10,
            "base": "weights/vgg11_base_model.pth",
            "expert": f"weights/vgg11_bn_expert_class_{FORGET_CLASS}.pth",
        },
        {
            "name": "MobileNetV2",
            "class": MobileNet_CIFAR10,
            "base": "weights/mobilenet_base_model.pth",
            "expert": f"weights/mobilenetv2_expert_class_{FORGET_CLASS}.pth",
        }
    ]

    results = {}

    for config in configs:
        arch_name = config['name']
        print(f"Analyzing collateral damage for: {arch_name}")

        if not all(os.path.exists(f) for f in [config['base'], config['expert']]):
            print(f"Missing weights for {arch_name}. Skipping.")
            continue

        base_model = config['class']().to(device)
        base_model.load_state_dict(torch.load(config['base'], map_location=device))
        
        expert_model = config['class']().to(device)
        expert_model.load_state_dict(torch.load(config['expert'], map_location=device))

        params = optimal_params[arch_name]
        unlearned_model = TaskArithmeticSurgeon.unlearn(
            base_model, expert_model, alpha=params['alpha'], drop_percentile=params['drop']
        )

        print("Measuring accuracy isolated on Airplane vs Other Classes")
        acc_airplane, acc_others = evaluate_overforgetting(
            unlearned_model, loaders['test_all'], device, FORGET_CLASS, NEIGHBOR_CLASS
        )

        print(f"Accuracy on Airplane (Neighbor): {acc_airplane:.2f}%")
        print(f"Accuracy on Other 8 Classes: {acc_others:.2f}%")
        
        gap = acc_others - acc_airplane
        print(f"Collateral Damage (Gap): {gap:.2f}%")

        results[arch_name] = {
            "Acc_Airplane": acc_airplane,
            "Acc_Others": acc_others,
            "Gap": gap
        }
        
        del base_model, expert_model, unlearned_model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/overforgetting_analysis.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main() 