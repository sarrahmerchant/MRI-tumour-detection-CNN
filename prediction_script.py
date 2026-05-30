from __future__ import annotations
from LLMExplanation import generate_plain_language_explanation

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_grad_cam
from PIL import Image
import cv2
import numpy as np


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_class_names(training_dir: Path) -> list[str]:
    ds = datasets.ImageFolder(root=str(training_dir))
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]


def build_model(num_classes: int, device: torch.device) -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model.eval()
    return model


def preprocess_image(img: Image.Image) -> torch.Tensor:
    pipeline = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if img.mode != "RGB":
        img = img.convert("RGB")
    return pipeline(img).unsqueeze(0)  # (1, 3, 224, 224)


def show_cam_on_image(img: Image.Image, grayscale_cam: np.ndarray) -> Image.Image:
    # PIL -> RGB float array; OpenCV needs ndarray, not Image
    h, w = int(grayscale_cam.shape[0]), int(grayscale_cam.shape[1])
    img_rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    if img_rgb.shape[0] != h or img_rgb.shape[1] != w:
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap = cv2.applyColorMap(np.uint8(255 * np.asarray(grayscale_cam, dtype=np.float32)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)

    output_image = heatmap * 0.5 + img_rgb * 0.5
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    return Image.fromarray(output_image)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    image_path = repo_root / "data" / "Testing" / "glioma" / "Te-gl_2.jpg"
    weights_path = repo_root / "model" / "model.pth"
    training_dir = repo_root / "data" / "Training"

    device = get_device()
    class_names = load_class_names(training_dir)
    model = build_model(num_classes=len(class_names), device=device)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    img = Image.open(image_path)
    processed_image = preprocess_image(img).to(device)
    processed_image.requires_grad = True

    with torch.no_grad():
        logits = model(processed_image)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        pred_idx = int(torch.argmax(probs).item())

    print(f"Device: {device}")
    print(f"Predicted: {class_names[pred_idx]} (p={probs[pred_idx].item():.4f})")
    print("All class probabilities:")
    for name, p in zip(class_names, probs.tolist()):
        print(f"  {name}: {p:.4f}")


    # get grad cam 
    target_layer = [model.layer4[-1]]
    cam = pytorch_grad_cam.GradCAM(model, target_layer)
    grayscale_cam = cam(input_tensor=processed_image, targets=None)
    grayscale_cam = grayscale_cam[0, :] # makes it flat?
    visualization = show_cam_on_image(img, grayscale_cam)
    visualization.save("grad_cam.png")

####### AI LLM Explanation #######

result = {
    "predicted_class": class_names[pred_idx],
    "confidence": probs[pred_idx].item(),
    "class_probabilities": {name: p.item() for name, p in zip(class_names, probs.tolist())},
    "explanation_signal": "Grad-CAM highlighted the region most influential to the prediction."
}

explanation = generate_plain_language_explanation(result)
print("\nAI Explanation:")
print(explanation)