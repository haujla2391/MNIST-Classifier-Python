import model
import data
import torch.nn as nn
import torch.optim as optim
import torch

def train():
    m = model.Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)

    train_loader, _ = data.get_dataloaders(64)

    optimizer = optim.Adam(m.parameters(), lr=0.001)  # Adam backprop takes parameters and learning rate (eta in weight update)
    criterion = nn.CrossEntropyLoss()                       # Computes the in sample Error for this multi class problem

    for epoch in range(5):
        total_loss = 0.0
        batch_count = 0

        m.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()                               # Resets gradients between batches so they aren't added together

            logits = m(images)                                           # same as calling m.forward(images)
            loss = criterion(logits, labels)

            loss.backward()                                     # backprop to compute gradients
            optimizer.step()                                    # Updates the weights

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}, Loss {avg_loss:.4f}")

    return m


# if __name__ == "__main__":
#     train()