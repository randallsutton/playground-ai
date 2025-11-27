import argparse
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from utils import build_model, get_data_transforms, get_device, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')

    parser.add_argument('data_dir', type=str, help='Directory containing train/valid/test subdirs')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'vgg13'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Hidden layer units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Set device
    if args.gpu:
        device = get_device()
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Set up data directories
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')

    # Load transforms and datasets
    data_transforms = get_data_transforms()
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
    }

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "valid": DataLoader(image_datasets["valid"], batch_size=64, shuffle=False),
    }

    # Build model
    num_classes = len(image_datasets["train"].classes)
    model = build_model(args.arch, args.hidden_units, num_classes)
    model.to(device)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Training loop
    print(f"Training {args.arch} for {args.epochs} epochs...")
    print(f"Total training batches: {len(dataloaders['train'])}")

    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        for inputs, labels in dataloaders["train"]:
            print(f"Running training at step {steps}...")

            batch_start = time.time()
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f"Device = {device}; Time per batch: {(time.time() - batch_start):.3f} seconds")

            if steps % print_every == 0:
                print(f"Running validation at step {steps}...")

                validation_start = time.time()

                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in dataloaders["valid"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Device = {device}; Time per validation: {(time.time() - validation_start):.3f} seconds")

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()

        print(f"Epoch {epoch+1} completed in {(time.time() - epoch_start):.3f} seconds")

    # Save checkpoint
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth')
    save_checkpoint(model, image_datasets["train"], args.epochs, checkpoint_path, args.arch)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
