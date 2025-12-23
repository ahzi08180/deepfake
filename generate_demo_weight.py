import torch
import torch.nn as nn
from torchvision import models

model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(1280, 1)

torch.save(model.state_dict(), "saved_models/demo_model.pth")
print("Demo model generated.")
