import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            _, predicted = torch.max(outputs, 1)

            total += y.size(0)

            correct += (predicted == y).sum().item()

    accuracy = correct / total

    return accuracy

def count_params(model): 
    return sum(p.numel() for p in model.parameters())

def get_confusion_matrix(model, test_loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return confusion_matrix(all_labels, all_preds)


def plot_confusion_matrix(cm, classes, ax, color, title="Confusion Matrix"):

    sns.heatmap(
        cm,
        annot=False,
        cmap=color,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

