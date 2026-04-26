import os
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.surgeon import TaskArithmeticSurgeon
from unlearning.utils import seed_everything

def get_entropy(model, dataloader, device, target_class):
    model.eval()
    entropies = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            mask = (labels == target_class)
            if mask.sum() == 0:
                continue
                
            inputs = inputs[mask]
            
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropies.extend(entropy.cpu().numpy())
            
    return np.array(entropies)

def perform_mia(model, train_loader, test_loader, device, target_class):
    # perform membership inference attack
    entropy_seen = get_entropy(model, train_loader, device, target_class)
    
    entropy_unseen = get_entropy(model, test_loader, device, target_class)
    
    y_true = np.concatenate([np.ones(len(entropy_seen)), np.zeros(len(entropy_unseen))])
    
    y_scores = np.concatenate([-entropy_seen, -entropy_unseen])
    
    auc = roc_auc_score(y_true, y_scores)
    return auc

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Starting MIA (Membership Inference Attack) on {device}")

    FORGET_CLASS = 8 # Ship
    loaders, class_names = get_cifar10_dataloaders(batch_size=256, forget_class=FORGET_CLASS)

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
        print(f"Running mia on: {arch_name}")

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

        auc_base = perform_mia(base_model, loaders['base_train'], loaders['test_all'], device, FORGET_CLASS)
        auc_unlearned = perform_mia(unlearned_model, loaders['base_train'], loaders['test_all'], device, FORGET_CLASS)

        print(f"Base Model MIA Vulnerability (AUC): {auc_base:.4f}")
        print(f"Unlearned Model Privacy (AUC): {auc_unlearned:.4f}")
        
        results[arch_name] = {
            "MIA_AUC_Base": auc_base,
            "MIA_AUC_Unlearned": auc_unlearned
        }
        
        del base_model, expert_model, unlearned_model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    os.makedirs("results", exist_ok=True)
    save_path = "results/mia_evaluation.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()