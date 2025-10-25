import time, statistics
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

MODEL_PATH = "resnet18_cifar10.pt"
N_IMAGES   = 200

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
loader   = DataLoader(test_ds, batch_size=1, shuffle=False)

def load_model(device="cpu"):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 10)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return m.to(device).eval()

@torch.no_grad()
def bench(device="cpu"):
    m = load_model(device)
    dummy = torch.randn(1,3,32,32).to(device)
    for _ in range(10): _ = m(dummy)  # 预热
    lat=[]
    for i,(x,_) in enumerate(loader):
        x = x.to(device)
        t0 = time.perf_counter(); _ = m(x); dt=(time.perf_counter()-t0)*1000
        lat.append(dt)
        if i+1>=N_IMAGES: break
    avg=sum(lat)/len(lat); p50=statistics.median(lat); p95=sorted(lat)[int(len(lat)*0.95)-1]; fps=1000/avg
    print(f"[{device}] n={len(lat)} avg={avg:.1f}ms p50={p50:.1f}ms p95={p95:.1f}ms FPS={fps:.1f}")

if __name__=="__main__":
    bench("cpu")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        bench("mps")
