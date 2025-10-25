import torch, torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 128
EPOCHS = 5               
LR = 1e-3
DEVICE = "cpu"           


train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])

def build_loaders():
    train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256,     shuffle=False, num_workers=0)
    return train_loader, test_loader

def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 预训练权重
    m.fc = nn.Linear(m.fc.in_features, 10)                        # 改成 10 类
    return m.to(DEVICE)

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total=0; correct=0; loss_sum=0.0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    print(f"Epoch {epoch}: train loss {loss_sum/total:.4f}, acc {correct/total*100:.2f}%")

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total=0; correct=0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    acc = correct/total*100
    print(f"Test accuracy: {acc:.2f}%")
    return acc

def main():
    train_loader, test_loader = build_loaders()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS+1):
        train_one_epoch(model, train_loader, criterion, optimizer, ep)

    acc = evaluate(model, test_loader)
    torch.save(model.state_dict(), "resnet18_cifar10.pt")
    print("Saved: resnet18_cifar10.pt")

if __name__ == "__main__":
    main()
