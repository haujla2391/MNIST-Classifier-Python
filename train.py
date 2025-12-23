import model
import data
import torch.nn as nn
import torch.optim as optim
import torch

def train(batch_size, lr, epochs):
    m = model.Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)

    train_loader, test_loader = data.get_dataloaders(batch_size)

    optimizer = optim.Adam(m.parameters(), lr=lr)  # Adam backprop takes parameters and learning rate (eta in weight update)
    criterion = nn.CrossEntropyLoss()                       # Computes the in sample Error for this multi class problem

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_loss = 0.0                            # compute losses in the train data for this epoch
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

            train_loss += loss.item()
            batch_count += 1

        avg_train_loss = train_loss / batch_count
        train_losses.append(avg_train_loss)

        m.eval()                                # compute losses in the test data for this epoch
        test_loss = 0.0
        test_batches = 0

        with torch.no_grad(): 
            for images, labels in test_loader:          
                images = images.to(device)
                labels = labels.to(device)

                logits = m(images)
                loss = criterion(logits, labels)

                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        test_losses.append(avg_test_loss)

    return m, train_losses, test_losses