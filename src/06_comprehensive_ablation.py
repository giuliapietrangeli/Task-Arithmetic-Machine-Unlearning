import os
import json
import torch
import copy
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.surgeon import TaskArithmeticSurgeon
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

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Comprehensive Ablation Study on {device}")

    loaders, class_names = get_cifar10_dataloaders(batch_size=1000, forget_class=0)
    print("Preloading test dataset")
    test_inputs, test_targets = [], []
    for x, y in loaders['test_all']:
        test_inputs.append(x)
        test_targets.append(y)
    test_inputs = torch.cat(test_inputs, dim=0).to(device)
    test_targets = torch.cat(test_targets, dim=0).to(device)

    # best parameters found - da task arithmetic 2 versione!
    optimal_params = {
        "ResNet18": {"alpha": -1.0, "drop": 0.629}, #arrotondato
        "VGG11_bn": {"alpha": -2.5, "drop": 0.990},
        "MobileNetV2": {"alpha": -1.5, "drop": 0.500}
    }

    # qui le arch
    architectures = [
        {"name": "ResNet18", "class": ResNetClassifier, "base": "weights/resnet18_base_model.pth"},
        {"name": "VGG11_bn", "class": VGG11_CIFAR10, "base": "weights/vgg11_base_model.pth"},
        {"name": "MobileNetV2", "class": MobileNet_CIFAR10, "base": "weights/mobilenet_base_model.pth"}
    ]

    master_results = {}

    for arch in architectures:
        arch_name = arch["name"]
        print(f"Running generalization test for {arch_name}")
        params = optimal_params[arch_name]
        print(f"Alpha: {params['alpha']} | Drop: {params['drop']}")

        if not os.path.exists(arch['base']):
            print(f"Base model missing for {arch_name}.")
            continue

        master_results[arch_name] = {"per_class": {}, "average": {}}
        
        total_retain = 0.0
        total_forget = 0.0
        classes_tested = 0

        base_model = arch['class']().to(device)
        base_model.load_state_dict(torch.load(arch['base'], map_location=device))

        pbar = tqdm(range(10), desc="Testing all 10 classes")
        for target_class in pbar:
            expert_path = f"weights/{arch_name.lower()}_expert_class_{target_class}.pth"
            
            if not os.path.exists(expert_path):
                print(f"Missing expert for class {target_class}.")
                continue

            expert_model = arch['class']().to(device)
            expert_model.load_state_dict(torch.load(expert_path, map_location=device))

            forget_mask = (test_targets == target_class)
            retain_mask = (test_targets != target_class)

            unlearned_model = TaskArithmeticSurgeon.unlearn(
                base_model, expert_model, 
                alpha=params['alpha'], 
                drop_percentile=params['drop']
            )

            acc_forget, acc_retain = evaluate_unlearning(
                unlearned_model, test_inputs, test_targets, forget_mask, retain_mask
            )
            class_name = class_names[target_class]
            master_results[arch_name]["per_class"][class_name] = {
                "retain": acc_retain,
                "forget": acc_forget
            }
            
            total_retain += acc_retain
            total_forget += acc_forget
            classes_tested += 1

            pbar.set_postfix({'Retain': f"{acc_retain:.1f}%", 'Forget': f"{acc_forget:.1f}%"})

            del expert_model, unlearned_model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        if classes_tested > 0:
            avg_retain = total_retain / classes_tested
            avg_forget = total_forget / classes_tested
            master_results[arch_name]["average"] = {"retain": avg_retain, "forget": avg_forget}
            print(f"Generalization results for {arch_name}:")
            print(f"   Average Retain: {avg_retain:.2f}% | Average Forget: {avg_forget:.2f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/comprehensive_ablation.json", "w") as f:
        json.dump(master_results, f, indent=4)

if __name__ == "__main__":
    main()