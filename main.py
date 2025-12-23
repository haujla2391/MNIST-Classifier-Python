from train import train
from eval import evaluate
import matplotlib.pyplot as plt

def main():
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5

    model, train_losses, test_losses = train(batch_size, learning_rate, num_epochs)

    accuracy = evaluate(model, batch_size)

    print(f"Accuracy: {accuracy:.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(len(test_losses)), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()