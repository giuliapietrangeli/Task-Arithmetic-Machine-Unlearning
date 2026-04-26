import torch
import numpy as np
import json
import os
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

def run_task_arithmetic_study():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    FORGET_CLASS = 8  
    
    loaders, class_names = get_cifar10_dataloaders(batch_size=1000, forget_class=FORGET_CLASS)
    print(f"Preloading test dataset (Target: {class_names[FORGET_CLASS]})")
    
    test_inputs, test_targets = [], []
    for x, y in loaders['test_all']:
        test_inputs.append(x)
        test_targets.append(y)
    test_inputs = torch.cat(test_inputs, dim=0).to(device)
    test_targets = torch.cat(test_targets, dim=0).to(device)
    
    forget_mask = (test_targets == FORGET_CLASS)
    retain_mask = (test_targets != FORGET_CLASS)

    # search space
    # alphas_to_test = [-0.5, -1.0, -1.5, -2.0] - this generated task_arithmetic_study - 1.json - VGG scelto alpha = -2.0
    alphas_to_test = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0] # this generated task_arithmetic_study.json - VGG scelto alpha = -2.5
    percentiles_to_test = np.linspace(0.50, 0.99, 20) 

    configs = [
        {
            "name": "ResNet18", 
            "class": ResNetClassifier, 
            "base": "weights/resnet18_base_model.pth", 
            "expert": f"weights/resnet18_expert_class_{FORGET_CLASS}.pth"
        },
        {
            "name": "VGG11_bn", 
            "class": VGG11_CIFAR10, 
            "base": "weights/vgg11_base_model.pth", 
            "expert": f"weights/vgg11_bn_expert_class_{FORGET_CLASS}.pth"
        },
        {
            "name": "MobileNetV2", 
            "class": MobileNet_CIFAR10, 
            "base": "weights/mobilenet_base_model.pth", 
            "expert": f"weights/mobilenetv2_expert_class_{FORGET_CLASS}.pth"
        }
    ]

    final_results = {}

    for config in configs:
        print(f"RIGOROUS TUNING FOR: {config['name']}")
        
        if not os.path.exists(config['expert']):
            print(f"Expert model missing: {config['expert']}. Skipping architecture.")
            continue

        base = config['class']().to(device)
        base.load_state_dict(torch.load(config['base'], map_location=device))
        
        expert = config['class']().to(device)
        expert.load_state_dict(torch.load(config['expert'], map_location=device))

        best_strict_score = -float('inf')
        best_fallback_score = -float('inf')
        
        best_strict = {"status": "FAILED"}
        best_fallback = {"status": "BEST_EFFORT"}

        for alpha in alphas_to_test:
            for drop in tqdm(percentiles_to_test, leave=False, desc=f"Testing Alpha {alpha}"):
                unlearned_model = TaskArithmeticSurgeon.unlearn(base, expert, alpha=alpha, drop_percentile=drop)
                acc_forget, acc_retain = evaluate_unlearning(unlearned_model, test_inputs, test_targets, forget_mask, retain_mask)
                
                score = acc_retain - (acc_forget * 1.5)
                
                if score > best_fallback_score: # miglior trade-off tra retain e forget
                    best_fallback_score = score
                    best_fallback = {
                        "alpha": alpha, 
                        "drop": drop, 
                        "retain": acc_retain, 
                        "forget": acc_forget,
                        "status": "BEST_EFFORT"
                    }
                
                if acc_retain > 75.0 and acc_forget < 20.0: # retain > 75% & forget < 20%
                    if score > best_strict_score:
                        best_strict_score = score
                        best_strict = {
                            "alpha": alpha, 
                            "drop": drop, 
                            "retain": acc_retain, 
                            "forget": acc_forget,
                            "status": "PASSED"
                        }

        if best_strict["status"] == "PASSED":
            print(f"STRICT PASSED: Alpha {best_strict['alpha']} | Drop {best_strict['drop']:.3f} | Retain {best_strict['retain']:.2f}% | Forget {best_strict['forget']:.2f}%")
            final_results[config['name']] = best_strict
        else:
            print(f"STRICT FAILED.")
            print(f"BEST EFFORT FOUND: Alpha {best_fallback['alpha']} | Drop {best_fallback['drop']:.3f} | Retain {best_fallback['retain']:.2f}% | Forget {best_fallback['forget']:.2f}%")
            final_results[config['name']] = best_fallback

    # save results
    os.makedirs("results", exist_ok=True)
    with open("results/task_arithmetic_study.json", "w") as f:
        json.dump(final_results, f, indent=4)
    
if __name__ == "__main__":
    run_task_arithmetic_study()