import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.surgeon import TaskArithmeticSurgeon
from unlearning.utils import seed_everything

def calculate_kl_divergence(unlearned_model, comparison_model, dataloader, device, target_class, mode='forget'):
    unlearned_model.eval()
    comparison_model.eval()
    
    total_kl = 0.0 # Divergenza KL
    samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if mode == 'forget':
                mask = (labels == target_class)
            else:
                mask = (labels != target_class)
                
            if mask.sum() == 0:
                continue
                
            inputs = inputs[mask]

            logits_unlearned = unlearned_model(inputs)
            logits_comparison = comparison_model(inputs)
            
            p_probs = F.softmax(logits_comparison, dim=1)
            q_log_probs = F.log_softmax(logits_unlearned, dim=1)
            
            kl = F.kl_div(q_log_probs, p_probs, reduction='batchmean')
            total_kl += kl.item() * inputs.size(0)
            samples += inputs.size(0)
            
    return total_kl / samples if samples > 0 else 0

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Starting Privacy Evaluation (KL Divergence) on {device}")

    FORGET_CLASS = 8 # Ship
    loaders, class_names = get_cifar10_dataloaders(batch_size=256, forget_class=FORGET_CLASS)
    print(f"Target Class: {class_names[FORGET_CLASS]}")

    # best parameters found - task arithmetic 2 vers come in 06.
    optimal_params = {
        "ResNet18": {"alpha": -1.0, "drop": 0.629},
        "VGG11_bn": {"alpha": -2.5, "drop": 0.990},
        "MobileNetV2": {"alpha": -1.5, "drop": 0.500}
    }

    # architectures
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
        print(f" Evaluating Privacy for: {arch_name}")

        if not all(os.path.exists(f) for f in [config['base'], config['expert'], config['comparison']]):
            print(f"Missing weights for {arch_name}. Skipping.")
            continue

        base_model = config['class']().to(device)
        base_model.load_state_dict(torch.load(config['base'], map_location=device))
        
        expert_model = config['class']().to(device)
        expert_model.load_state_dict(torch.load(config['expert'], map_location=device))
        
        comparison_model = config['class']().to(device)
        comparison_model.load_state_dict(torch.load(config['comparison'], map_location=device))

        params = optimal_params[arch_name]
        print(f"Applying Surgery -> Alpha: {params['alpha']} | Drop: {params['drop']}")
        unlearned_model = TaskArithmeticSurgeon.unlearn(
            base_model, expert_model, alpha=params['alpha'], drop_percentile=params['drop']
        )

        print("Calculating KL Divergence on Forget Data (Ship)")
        kl_div_forget = calculate_kl_divergence(
            unlearned_model, comparison_model, loaders['test_all'], device, FORGET_CLASS, mode='forget'
        )
        
        print("Calculating KL Divergence on Retain Data")
        kl_div_retain = calculate_kl_divergence(
            unlearned_model, comparison_model, loaders['test_all'], device, FORGET_CLASS, mode='retain'
        )

        print(f"Results for {arch_name}:")
        print(f"KL Divergence (Forget Class): {kl_div_forget:.4f}")
        print(f"KL Divergence (Retain Class): {kl_div_retain:.4f}")

        results[arch_name] = {
            "KL_Forget": kl_div_forget,
            "KL_Retain": kl_div_retain
        }
        
        del base_model, expert_model, comparison_model, unlearned_model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/privacy_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()