import torch
import cv2
import matplotlib.pyplot as plt

def generate_CAM(img_path, model, layer_name='layer4'):
    """
    Generate the CAM image for given input image and model.

    Parameters:
    - img_path: path to the input image
    - model: trained model
    - layer_name: name of the layer to use for CAM generation

    Returns:
    - CAM image
    """

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    inputs = transform(img).unsqueeze(0)

    # Placeholder for the feature maps
    feature_maps = []

    # Hook to extract the feature maps
    def hook_fn(module, input, output):
        feature_maps.append(output)

    # Register the hook
    getattr(model, layer_name).register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    # Get the weights of the fully connected layer (fc)
    params = list(model.parameters())
    fc_weights = params[-2]

    # Compute CAM
    cam = torch.matmul(fc_weights, feature_maps[0].reshape(feature_maps[0].shape[1], -1))
    cam = cam.reshape(outputs.shape[-1], 224, 224)
    cam = cam - torch.min(cam)
    cam_img = cam / torch.max(cam)
    cam_img = torch.nn.functional.interpolate(cam_img, scale_factor=(32, 32), mode='bilinear')

    # Convert to numpy and apply colormap
    cam_img = cam_img.squeeze().cpu().numpy()
    cam_img = cv2.applyColorMap(np.uint8(255 * cam_img), cv2.COLORMAP_JET)
    cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)

    return cam_img

# For demonstration purposes, we'll use a dummy model
# TODO: Replace with actual model loading and weight loading
# model = ResNet_CBAM_NonLocal(...)
# model.load_state_dict(torch.load('path_to_weights.pth'))

# Let's assume you provided a path to an image named "sample.jpg"
# CAM_img = generate_CAM("sample.jpg", model)

# plt.imshow(CAM_img)
# plt.show()
