from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from driftguard.model.c_resnet.model import get_cresnet
from driftguard.model.c_vit.model import get_cvit
from driftguard.model.dataset import get_train_transform
from driftguard.model.training.trainer import Trainer, TrainConfig
import torch.nn as nn

root = "datasets/init_subsets/pacs_art_painting"
tfm = get_train_transform(224)
ds = datasets.ImageFolder(root, transform=tfm)
loader = DataLoader(ds, batch_size=10, shuffle=False)

model = get_cresnet(num_classes=10, layers=[1,1,1,1])
# model = get_cvit(num_classes=10, image_size=224,patch_size=16)

trainer = Trainer(
    model,
    loss_fn=nn.CrossEntropyLoss(),
    config=TrainConfig(epochs=100, device="cpu", accumulate_steps=3),
)
# history = trainer.fit(loader, None)
metrix, l1_w, l2_w, softs = trainer.inference(loader)
print(softs.shape, l2_w.shape)



