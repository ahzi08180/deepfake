import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        """
        input_tensor: shape (1, C, H, W)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        # binary classifier (sigmoid)
        if output.dim() == 2 and output.size(1) == 1:
            score = output[0, 0]
        else:
            score = output.max()

        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam[0].cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam


def overlay_cam(image_pil, cam):
    """
    image_pil: PIL.Image (原圖)
    cam: numpy array (H, W) from Grad-CAM
    """

    # PIL -> numpy
    image = np.array(image_pil)

    # 將 CAM resize 成 image 大小
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

    # CAM -> heatmap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 疊圖
    overlay = heatmap * 0.4 + image * 0.6
    return overlay.astype(np.uint8)