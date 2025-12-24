import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: your CNN model
        target_layer: the last convolutional layer (nn.Module)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: (1, C, H, W)
        target_class: None -> use predicted class
        """
        self.model.zero_grad()

        output = self.model(input_tensor)  # shape: (1, num_classes) or (1, 1)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # For binary classifier (sigmoid output)
        if output.shape[-1] == 1:
            score = output[0]
        else:
            score = output[0, target_class]

        score.backward(retain_graph=True)

        # Global Average Pooling on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam[0].cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam
