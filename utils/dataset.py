import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(image_size=128, gray_scale=False):
    """
    Возвращает преобразования для изображений
    """
    if gray_scale:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    return transform

def get_datasets(data_dir="data", image_size=128, gray_scale=False):
    """
    Загружает train и test датасеты
    """
    transform = get_transforms(image_size, gray_scale)

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

def get_dataloaders(data_dir="data", batch_size=32, image_size=128, gray_scale=False):
    """
    Возвращает train и test обьекты DataLoader
    """
    train_dataset, test_dataset = get_datasets(data_dir, image_size, gray_scale)

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

from PIL import Image
import os

def get_dataset_info(data_dir):
    """Возвращает информацию о датасете: количество каналов, высоту и ширину изображений"""
    train_path = os.path.join(data_dir, "train")

    class_name = os.listdir(train_path)[0]
    class_path = os.path.join(train_path, class_name)

    img_name = os.listdir(class_path)[0]
    img_path = os.path.join(class_path, img_name)

    img = Image.open(img_path)

    width, height = img.size

    channels = len(img.getbands())

    return {
        "channels": channels,
        "height": height,
        "width": width
    }