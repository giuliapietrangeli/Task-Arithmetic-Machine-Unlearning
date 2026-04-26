from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_cifar10_dataloaders(batch_size=128, forget_class=1):
    # data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), #augm. - crop
        transforms.RandomHorizontalFlip(), #augm. - filp
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) # downloads only if not exists
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    #splits (liste indici per creare subset)
    forget_train_idx = [i for i, target in enumerate(train_dataset.targets) if target == forget_class]
    forget_train_ds = Subset(train_dataset, forget_train_idx)
    forget_test_idx = [i for i, target in enumerate(test_dataset.targets) if target == forget_class]
    retain_test_idx = [i for i, target in enumerate(test_dataset.targets) if target != forget_class]
    forget_test_ds = Subset(test_dataset, forget_test_idx)
    retain_test_ds = Subset(test_dataset, retain_test_idx)
    
    # dataloaders
    loaders = {
        'base_train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        'forget_train': DataLoader(forget_train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        'test_all': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
        'test_forget': DataLoader(forget_test_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        'test_retain': DataLoader(retain_test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    }
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"Data loaders ready")
    return loaders, class_names
