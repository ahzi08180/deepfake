import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class DeepfakeImageModel:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = nn.Linear(1280, 1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, face_img):
        img = Image.fromarray(face_img[:, :, ::-1])
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).item()

        return prob
