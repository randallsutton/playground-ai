import argparse
import json
import torch
from utils import get_device, load_checkpoint, predict


def main():
    parser = argparse.ArgumentParser(description='Predict flower class from an image')

    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint file')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Set device
    if args.gpu:
        device = get_device()
    else:
        device = torch.device("cpu")

    # Load model
    model = load_checkpoint(args.checkpoint_path)
    model.to(device)

    # Load category names if provided
    cat_to_name = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    # Run prediction
    probs, classes = predict(args.image_path, model, args.top_k)

    # Print results
    for i, (prob, cls) in enumerate(zip(probs, classes)):
        name = cat_to_name[cls] if cat_to_name else cls
        print(f"{i+1}. {name}: {prob:.4f}")


if __name__ == "__main__":
    main()
