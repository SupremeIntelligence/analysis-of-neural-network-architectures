import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(image_size=128):
    """
    Возвращает преобразования для изображений
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],  
            std=[0.5]
        )
    ])
    return transform

def get_datasets(data_dir="data", image_size=128):
    """
    Загружает train и test датасеты
    """
    transform = get_transforms(image_size)

    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=transform
    )

    return train_dataset, test_dataset

def get_dataloaders(data_dir="data", batch_size=32, image_size=28):
    """
    Возвращает train и test обьекты DataLoader
    """
    train_dataset, test_dataset = get_datasets(data_dir, image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

def get_class_names(data_dir="data"):
    """
    Возвращает список классов
    """
    train_path = os.path.join(data_dir, "train")
    dataset = datasets.ImageFolder(root=train_path)
    return dataset.classes