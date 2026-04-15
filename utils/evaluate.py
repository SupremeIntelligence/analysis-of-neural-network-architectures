import torch

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