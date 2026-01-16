from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split, Subset
import torch
import torch.nn as nn

from driftguard.model.training.trainer import TrainConfig, Trainer
from driftguard.model.vit import ViTArgs, VisonTransformer


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
    embed_dim=128,
    embed_image_size=224,
    embed_patch_size=16,
    encoder_depth=4,
    mha_num_heads=4,
    repr_dim=128,
    head_num_classes=7,
)
model = VisonTransformer(args)

model = resnet18(num_classes=7)
trainer = Trainer(
    model,
    loss_fn=nn.CrossEntropyLoss(),
    config=TrainConfig(epochs=10, device="cuda", amp=True, accumulate_steps=2),
)
history = trainer.fit(train_loader, val_loader)
