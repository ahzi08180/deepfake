import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class RVF10KDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        for label, cls in enumerate(["real", "fake"]):
            folder = os.path.join(root_dir, split, cls)
            for f in os.listdir(folder):
                self.samples.append((os.path.join(folder, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
