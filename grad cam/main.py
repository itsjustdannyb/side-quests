import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import models
import torchvision.transforms.v2 as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get the class names from the weights metadata
weights = models.ResNet50_Weights.DEFAULT
classes = weights.meta["categories"]

# Create model using the same weights
model = models.resnet50(weights=weights).to(device)

# load image
transform = T.Compose([
    T.ToImage(),
    T.Resize((224,224)),
    T.ToDtype(torch.float)
])

image = Image.open(r"image.png")
image_tans = transform(image)[:3, :, :].unsqueeze(0).to(device)

## grad cam
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.target_layer = target_layer
        self.handles = []

        self._register_hooks()

    def get_activations(self, module, input, output):
        self.activations = output
        print(f"caught activations")

    def get_gradients(self, module, input, output):
        self.gradients = output[0]
        print(f"caught gradients")

    def _register_hooks(self):
        self.handles.append(self.target_layer.register_forward_hook(self.get_activations))
        self.handles.append(self.target_layer.register_full_backward_hook(self.get_gradients))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

target_layer = model.layer4[2].conv3
grad_cam = GradCAM(model, target_layer)


# forward pass
model.eval()
output = model(image_tans)

# backward pass
pred_idx = torch.argmax(output, dim=1).item()
target = output[0, pred_idx]
target.backward()


# get gradients
gradients = grad_cam.gradients
activations = grad_cam.activations

# pool gradients across height and widrth
weights = torch.mean(gradients, dim=(2,3), keepdim=True)

# weight the activations
weighted_activations = torch.mul(gradients, weights)

# sum across the channels
heatmap = torch.sum(weighted_activations, dim=1)

# apply relu
heatmap = F.relu(heatmap)

# process for visualization
heatmap = heatmap.detach().cpu().numpy()[0]

if np.max(heatmap) == 0:
    pass
else:
    heatmap /= np.max(heatmap)

resized_image = image.resize((224,224))
heatmap_image = np.array(Image.fromarray(255 * heatmap).resize((224,224)))

colored_map = plt.cm.jet(heatmap_image / 255.0)[:, :, :3]

# overlay heatmap on image
superposed_image = 0.6 * np.array(resized_image)[:, :, :3] / 255.0 + 0.4 * colored_map


plt.figure(figsize=(10,5))
plt.imshow(superposed_image)
plt.title(f"Prediction: {classes[pred_idx]}")
plt.axis("off")
plt.show()

grad_cam.remove_hooks()