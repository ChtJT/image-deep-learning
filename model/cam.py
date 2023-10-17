# Complete code for generating Grad-CAM heatmaps

import torch
import cv2
import numpy as np
import os
import glob
from torchvision import transforms
from PIL import Image
from ResNet_CBAM_NonLocal import resnet34_CBAM_NonLocal

class GradCAM:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.feature_map = None
        self.gradient = None
        self.hook_a = self.feature_layer.register_forward_hook(self._hook_fn_a)
        self.hook_b = self.feature_layer.register_backward_hook(self._hook_fn_b)

    def _hook_fn_a(self, module, input, output):
        self.feature_map = output.detach()

    def _hook_fn_b(self, module, grad_in, grad_out):
        self.gradient = grad_out[0].detach()

    def forward(self, x):
        return self.model(x)

    def backward_on_target(self, output, target):
        self.model.zero_grad()
        one_hot_output = torch.zeros([1, output.size()[-1]])
        one_hot_output[0][target] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

    def generate(self, input_tensor):
        output = self.forward(input_tensor)

        # Perform backward pass
        target_class = output.argmax().item()
        self.backward_on_target(output, target_class)

        # Same code as before to generate CAM
        feature_map = self.feature_map[0].cpu().numpy()
        gradient = self.gradient[0].cpu().numpy()
        weights = np.mean(gradient, axis=(1, 2))
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feature_map[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def overlay_heatmap(cam, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    cam = 1 - cam  # Inverting the CAM values
    cam = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)


def generate_heatmaps_for_model(model, model_weights_path, feature_layer, source_dir, target_dir):
    """
    Generate Grad-CAM heatmaps for images in source_dir and save to target_dir.

    Args:
    - model: Initialized PyTorch model.
    - model_weights_path: Path to the .pth file with model weights.
    - feature_layer: The layer of the model to use for Grad-CAM.
    - source_dir: Directory with source images.
    - target_dir: Directory to save generated heatmaps.
    """
    # Load model weights
    model_weights = torch.load(model_weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_weights)
    model.eval()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    image_files = glob.glob(os.path.join(source_dir, '**', '*.*'), recursive=True)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for image_path in image_files:
        raw_image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(raw_image).unsqueeze(0)

        # Using GradCAM directly inside the loop to generate CAM
        grad_cam = GradCAM(model, feature_layer)
        cam = grad_cam.generate(input_tensor)

        result = overlay_heatmap(cam, np.array(raw_image.resize((224, 224))))

        save_path = os.path.join(target_dir, os.path.relpath(image_path, source_dir))
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        Image.fromarray(result).save(save_path)

model = resnet34_CBAM_NonLocal()
model.fc = torch.nn.Linear(model.fc.in_features, 3)
feature_layer = model.nl4
generate_heatmaps_for_model(model, "model_weights.pth", feature_layer, "data_split/train", "data_split/train_heatmaps")
