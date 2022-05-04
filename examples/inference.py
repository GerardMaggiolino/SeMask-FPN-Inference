"""
Load a model, run inference, and save an image in the dataset's
standard palette.
"""
import pdb

import torch
import semask
import torchvision
from torchvision import io


def main():
    # Parameters
    device = "cpu"
    img_name = "cityscapes.png"
    model_name = "cityscapes_base"

    # Load image
    img = io.read_image(img_name).to(torch.float32) / 255
    img = torchvision.transforms.functional.normalize(
        img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    img = img.to(device).unsqueeze(0)

    # Load model and get predictions
    model = semask.load_model(model_name).eval()
    model.to(device)
    with torch.no_grad():
        predictions = model.inference(img, rescale=True)
    predictions = predictions.squeeze(0).argmax(dim=0)

    # Check out model attributes
    for idx, class_name in enumerate(model.classes):
        print(f"{idx} {class_name} {model.palette[idx]}")

    colors = model.palette[predictions.view(-1)].permute((1, 0))
    viz_img = colors.view(3, img.shape[2], img.shape[3])
    viz_img = viz_img.to(torch.float32) / 255
    torchvision.utils.save_image(viz_img, f"inference_image.png")


if __name__ == "__main__":
    main()
