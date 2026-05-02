import os
import json
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.surgeon import TaskArithmeticSurgeon
from unlearning.utils import seed_everything

def calculate_zrf_score(unlearned_model, incompetent_model, dataloader, device, target_class):
    unlearned_model.eval()
    incompetent_model.eval()
    
    total_js = 0.0 #js score
    samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            mask = (labels == target_class)
            if mask.sum() == 0:
                continue
                
            inputs = inputs[mask]
            
            logits_unl = unlearned_model(inputs)
            p_probs = F.softmax(logits_unl, dim=1)
            
            logits_inc = incompetent_model(inputs)
            q_probs = F.softmax(logits_inc, dim=1)
            
            m_probs = 0.5 * (p_probs + q_probs)
            
            kl_p_m = F.kl_div(torch.log(m_probs + 1e-8), p_probs, reduction='batchmean')
            kl_q_m = F.kl_div(torch.log(m_probs + 1e-8), q_probs, reduction='batchmean')
            
            js_div = 0.5 * (kl_p_m + kl_q_m)
            
            js_normalized = js_div.item() / math.log(2)
            
            total_js += js_normalized * inputs.size(0)
            samples += inputs.size(0)
            
    avg_js = total_js / samples if samples > 0 else 0
    
    zrf_score = 1.0 - avg_js
    return zrf_score

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Starting ZRF Score Evaluation on {device}")

    FORGET_CLASS = 8 # Ship
    loaders, class_names = get_cifar10_dataloaders(batch_size=256, forget_class=FORGET_CLASS) # dataloader per le varie classi

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
        print(f"Calculating ZRF score for {arch_name}")

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

        incompetent_model = config['class']().to(device)

        print("Probabilistic comparison: Unlearned Model vs Incompetent")
        zrf = calculate_zrf_score(unlearned_model, incompetent_model, loaders['test_all'], device, FORGET_CLASS)

        print(f"ZRF Score: {zrf:.4f}")

        results[arch_name] = {
            "ZRF_Score": zrf
        }
        
        del base_model, expert_model, unlearned_model, incompetent_model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/zrf_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()