import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from unlearning.model import ResNetClassifier, VGG11_CIFAR10, MobileNet_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.utils import seed_everything

def train_single_expert(model_class, base_weights_path, expert_save_path, forget_class, lr, device):
    #train expert 
    loaders, class_names = get_cifar10_dataloaders(batch_size=128, forget_class=forget_class)
    class_name = class_names[forget_class]
    
    print(f"    Training Expert for Class {forget_class} ({class_name})...")
    
    # load base model weights
    model = model_class().to(device)
    model.load_state_dict(torch.load(base_weights_path, map_location=device))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    # fine-tune on forget class only
    for epoch in range(5): 
        for inputs, labels in loaders['forget_train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    # Save expert weights
    torch.save(model.state_dict(), expert_save_path)
    
    #Per la memoria
    del model, loaders, optimizer, criterion
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs("weights", exist_ok=True)

    # architecture specific parameters
    configs = [
        {
            "name": "ResNet18",
            "class": ResNetClassifier,
            "base": "weights/resnet18_base_model.pth",
            "lr": 1e-4
        },
        {
            "name": "VGG11_bn",
            "class": VGG11_CIFAR10,
            "base": "weights/vgg11_base_model.pth",
            "lr": 5e-5
        },
        {
            "name": "MobileNetV2",
            "class": MobileNet_CIFAR10,
            "base": "weights/mobilenet_base_model.pth",
            "lr": 1e-4
        }
    ]

    print("Preparing to train 30 expert models")

    for config in configs:
        print(f"Architecture: {config['name']}")
        
        # train 10 experts per architecture
        for class_idx in range(10):
            save_path = f"weights/{config['name'].lower()}_expert_class_{class_idx}.pth"
            
            if os.path.exists(save_path):
                print(f"   Expert for Class {class_idx} already exists.")
                continue
                
            train_single_expert(
                model_class=config['class'],
                base_weights_path=config['base'],
                expert_save_path=save_path,
                forget_class=class_idx,
                lr=config['lr'],
                device=device
            )

if __name__ == "__main__":
    main()