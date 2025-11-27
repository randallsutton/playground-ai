import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torchvision import models, transforms


def build_model(arch, hidden_units, num_classes):
    """Creates a VGG model with a custom classifier."""
    if arch == "vgg13":
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
    else:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, num_classes),
        nn.LogSoftmax(dim=1)
    )

    return model


def display_prediction(image_path, model, cat_to_name, topk=5):
    """Display an image alongside a bar chart of the top-k predicted classes."""
    # Get predictions
    probs, classes = predict(image_path, model, topk)

    # Get flower names
    flower_names = [cat_to_name[cls] for cls in classes]

    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)

    # Display image
    with Image.open(image_path) as img:
        ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(flower_names[0])

    # Display bar chart (horizontal, with highest probability at top)
    y_pos = np.arange(len(flower_names))
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(flower_names)
    ax2.invert_yaxis()  # Highest probability at top

    plt.tight_layout()
    plt.show()


def get_data_transforms():
    """Returns data transforms for train, valid, and test sets."""
    return {
        "train": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }


def get_device():
    """Returns the best available device (MPS/CUDA/CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def imshow(image, ax=None, title=None):
    """Display a tensor as an image."""
    if ax is None:
        _, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes it's the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    if title:
        ax.set_title(title)

    return ax


def load_checkpoint(filepath):
    """Loads a saved model checkpoint and returns the model."""
    checkpoint = torch.load(filepath, weights_only=False)

    arch = checkpoint.get("arch", "vgg16")
    if arch == "vgg13":
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
    else:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    model.eval()

    return model


def predict(image_path, model, topk=5):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    with Image.open(image_path) as image:
        np_image = process_image(image)
        tensor_image = torch.from_numpy(np_image).float().unsqueeze(0)
        tensor_image = tensor_image.to(next(model.parameters()).device)

        model.eval()
        with torch.no_grad():
            output = model(tensor_image)

        # Get probabilities
        ps = torch.softmax(output, dim=1)

        # Get top-k probabilities and indices
        top_p, top_idx = ps.topk(topk, dim=1)

        # Convert to numpy arrays
        top_p = top_p.cpu().numpy().flatten()
        top_idx = top_idx.cpu().numpy().flatten()

        # Invert class_to_idx to get idx_to_class
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}

        # Map indices to class labels
        top_classes = [idx_to_class[idx] for idx in top_idx]

        return top_p, top_classes


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array.
    """
    # Resize so shortest side is 256, maintaining aspect ratio
    width, height = image.size
    ratio = 256 / min(width, height)
    image = image.resize((int(width * ratio), int(height * ratio)))

    # Center crop to 224x224
    crop_left = (image.width - 224) / 2
    crop_top = (image.height - 224) / 2
    image = image.crop((crop_left, crop_top, crop_left + 224, crop_top + 224))

    # Convert to numpy array, scale to 0-1, and normalize
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Set color channel to first dimension
    return np_image.transpose((2, 0, 1))


def save_checkpoint(model, train_dataset, epochs, filepath, arch="vgg16"):
    """Saves a model checkpoint."""
    checkpoint = {
        "arch": arch,
        "state_dict": model.state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
        "classifier": model.classifier,
        "epochs": epochs,
    }
    torch.save(checkpoint, filepath)
