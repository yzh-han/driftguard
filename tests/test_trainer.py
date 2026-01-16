from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split, Subset
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

from driftguard.model.training.trainer import TrainConfig, Trainer
from driftguard.model.vit import ViTArgs, VisonTransformer

# 看看模型大小
def model_size(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    mb = bytes_ / (1024**2)

    model_name = model.__class__.__name__
    print(f"{model_name} - 参数: total={total:,}, trainable={trainable:,}, approx={mb:.2f} MB")
    return total, trainable, mb

transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
dataset = datasets.ImageFolder("../datasets/pacs/photo", transform=transform)

total = 600
if len(dataset) < total:
    raise ValueError(f"dataset too small: {len(dataset)} < {total}")

perm = torch.randperm(len(dataset))[:total]
train_idx = perm[:500].tolist()
val_idx = perm[500:600].tolist()

train_loader = DataLoader(
    Subset(dataset, train_idx), batch_size=8, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    Subset(dataset, val_idx), batch_size=8, shuffle=False, num_workers=0
)

args = ViTArgs(
    embed_dim=256,
    embed_image_size=224,
    embed_patch_size=16,
    encoder_depth=6,
    mha_num_heads=8,
    repr_dim=256,
    head_num_classes=7,
)

model = VisonTransformer(args)
model_size(model)
trainer = Trainer(
    model,
    loss_fn=nn.CrossEntropyLoss(),
    config=TrainConfig(epochs=100, device="cuda", amp=True, accumulate_steps=1, lr=3e-4),
)
history = trainer.fit(train_loader, val_loader)
print()

model = vit_b_16(num_classes=7)
model_size(model)
trainer = Trainer(
    model,
    loss_fn=nn.CrossEntropyLoss(),
    config=TrainConfig(epochs=10, device="cuda", amp=True, accumulate_steps=2),
)
history = trainer.fit(train_loader, val_loader)
print()

model = resnet18(num_classes=7)
model_size(model)
trainer = Trainer(
    model,
    loss_fn=nn.CrossEntropyLoss(),
    config=TrainConfig(epochs=10, device="cuda", amp=True, accumulate_steps=2),
)
history = trainer.fit(train_loader, val_loader)
print()

