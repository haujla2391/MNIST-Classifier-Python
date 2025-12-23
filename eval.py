import data
import torch


def evaluate(model, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()            # switches off training-specific behavior (dropout, batchnorm)

    _, test_loader = data.get_dataloaders(batch_size)

    correct, total = 0, 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)            # softmax just produces values 0-1 for the same classes so argmax is same

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return acc