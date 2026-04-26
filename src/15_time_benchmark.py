import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from unlearning.model import VGG11_CIFAR10
from unlearning.dataset import get_cifar10_dataloaders
from unlearning.surgeon import TaskArithmeticSurgeon
from unlearning.utils import seed_everything

def execute_full_retraining(model, dataloader, device, epochs, target_class):
    # retrain model from scratch
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        for inputs, labels in tqdm(dataloader, desc=f"Retraining (Epoch {epoch}/{epochs})", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            mask = (labels != target_class)
            if mask.sum() == 0:
                continue
            inputs, labels = inputs[mask], labels[mask]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    end_time = time.time()
    return end_time - start_time

def main():
    seed_everything(42) #Seed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Efficiency Benchmark on {device}")

    FORGET_CLASS = 8 
    EPOCHS_ORIGINALI = 30 
    
    loaders, class_names = get_cifar10_dataloaders(batch_size=128, forget_class=FORGET_CLASS)

    base_model = VGG11_CIFAR10().to(device) # solo vgg - il più lento
    expert_model = VGG11_CIFAR10().to(device)
    retrain_model = VGG11_CIFAR10().to(device)
    
    print("Phase 1: Task Arithmetic Measurement")
    start_ta = time.time()
    unlearned_model = TaskArithmeticSurgeon.unlearn(base_model, expert_model, alpha=-2.5, drop_percentile=0.990)
    end_ta = time.time()
    time_ta = end_ta - start_ta
    print(f"Operation completed in {time_ta:.4f} seconds.")

    print(f"Phase 2: Full Retraining Execution ({EPOCHS_ORIGINALI} Epochs)")
    time_retrain_total = execute_full_retraining(retrain_model, loaders['base_train'], device, EPOCHS_ORIGINALI, FORGET_CLASS)
    print(f"Retraining completed in {time_retrain_total:.2f} seconds.")
    
    savings = (1.0 - (time_ta / time_retrain_total)) * 100
    
    print("Real Time Results")
    print(f"Task Arithmetic Time: {time_ta:.4f} sec")
    print(f"Retraining Time (30e): {time_retrain_total:.2f} sec")
    print(f"Computational Savings:{savings:.4f}%")

    results = {
        "VGG11_bn": {
            "Task_Arithmetic_Time_sec": time_ta,
            "Retraining_Time_sec": time_retrain_total,
            "Computational_Savings_Percent": savings
        }
    }
    
    # save results
    os.makedirs("results", exist_ok=True)
    with open("results/time_benchmark.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()