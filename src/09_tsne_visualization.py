import os

#soluzione per segmentation fault
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.surgeon import TaskArithmeticSurgeon
from unlearning.utils import seed_everything

def extract_features(model, dataloader, device, num_samples=1000):
    model.eval()
    all_features = []
    all_labels = []
    
    samples_collected = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_features.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            
            samples_collected += inputs.size(0)
            if samples_collected >= num_samples:
                break
                
    features = np.vstack(all_features)[:num_samples]
    labels = np.concatenate(all_labels)[:num_samples]
    return features, labels

def plot_tsne(base_features, unlearned_features, labels, target_class, save_path, title):
    
    base_features = base_features.astype(np.float64)
    unlearned_features = unlearned_features.astype(np.float64)
    print(f"  Calculating Dimensionality Reduction for Base Model")
    tsne_model = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', n_jobs=1)
    
    tsne_base = tsne_model.fit_transform(base_features)
    print(f"  Calculating Dimensionality Reduction for Unlearned Model")
    tsne_unlearned = tsne_model.fit_transform(unlearned_features)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Latent Space Analysis: {title}', fontsize=16)
    
    colors = ['#d3d3d3'] * 10
    colors[target_class] = '#e63946'  # Rosso
    
    alphas = [0.3] * 10
    alphas[target_class] = 1.0
    
    for i in range(10):
        idx = (labels == i)
        if i == target_class:
            ax1.scatter(tsne_base[idx, 0], tsne_base[idx, 1], c=colors[i], alpha=alphas[i], label=f'Target (Ship)', s=30, zorder=5)
        else:
            ax1.scatter(tsne_base[idx, 0], tsne_base[idx, 1], c=colors[i], alpha=alphas[i], s=10, zorder=1)
            
    ax1.set_title('Base Model (Intact Memory)')
    ax1.axis('off')
    ax1.legend()

    for i in range(10):
        idx = (labels == i)
        if i == target_class:
            ax2.scatter(tsne_unlearned[idx, 0], tsne_unlearned[idx, 1], c=colors[i], alpha=alphas[i], label=f'Target (Ship)', s=30, zorder=5)
        else:
            ax2.scatter(tsne_unlearned[idx, 0], tsne_unlearned[idx, 1], c=colors[i], alpha=alphas[i], s=10, zorder=1)
            
    ax2.set_title('Unlearned Model (Concept Destroyed)')
    ax2.axis('off')
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved in: {save_path}")

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Visualization on {device}")

    FORGET_CLASS = 8 # Ship
    loaders, class_names = get_cifar10_dataloaders(batch_size=64, forget_class=FORGET_CLASS)

    # best parameters - vedi 06
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

    for config in configs:
        arch_name = config['name']
        print(f"Generating plots for: {arch_name}")

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

        print("Extracting decision space features (1500 samples)")
        base_features, labels = extract_features(base_model, loaders['test_all'], device, num_samples=1500)
        unlearned_features, _ = extract_features(unlearned_model, loaders['test_all'], device, num_samples=1500)

        save_path = f"results/plots/tsne_{arch_name.lower()}.png"
        plot_tsne(base_features, unlearned_features, labels, FORGET_CLASS, save_path, arch_name)
        
        del base_model, expert_model, unlearned_model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    main()