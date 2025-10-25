import os, cv2, numpy as np, torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

os.makedirs("miscls", exist_ok=True)

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
ds = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
classes = ds.classes
loader = DataLoader(ds, batch_size=1, shuffle=False)

m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
m.fc = nn.Linear(m.fc.in_features, 10)
m.load_state_dict(torch.load("resnet18_cifar10.pt", map_location="cpu"))
m.eval()

def to_img(x: torch.Tensor) -> np.ndarray:
    # CHW(float32, normalized) -> HWC(uint8, RGB), 并确保C连续
    x = x[0].detach().cpu().numpy().transpose(1, 2, 0)   # HWC
    x = (x * STD + MEAN)                                 # 反标准化
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    x = np.ascontiguousarray(x)                          # 关键：保证 C-contiguous
    return x

saved = 0
with torch.no_grad():
    for i, (x, y) in enumerate(loader):
        logits = m(x)
        pred = logits.argmax(1).item()
        gt   = y.item()
        if pred != gt:
            img = to_img(x)
            
            txt = f"pred: {classes[pred]}   gt: {classes[gt]}"
            cv2.putText(img, txt, (2, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            out = f"miscls/sample_{i:04d}.jpg"
            cv2.imwrite(out, img)
            print("saved:", out, "->", txt)
            saved += 1
            if saved >= 3:
                break

print(f"Done. Saved {saved} samples into ./miscls/")
