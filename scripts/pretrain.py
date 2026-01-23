import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from driftguard.model.c_resnet.model import get_cresnet
from driftguard.model.c_vit.model import get_cvit
from driftguard.model.dataset import get_inference_transform, get_train_transform
from driftguard.model.training.trainer import Trainer, TrainConfig
from torch.utils.data import Subset
import torch.nn as nn
from torchvision.models import resnet18

for root, test_root, image_size, num_classes, d_name in[
    # ("datasets/init_subsets/dg5_mnist", "datasets/dg5/mnist", 28, 10,"dg5"),
    ("datasets/init_subsets/pacs_art_painting", "datasets/pacs/art_painting", 224,7, "pacs"),
    # ("datasets/init_subsets/domainnet_clipart", "datasets/drift_domain_net/clipart",224, 7, "ddn")
]:
    # root = "datasets/init_subsets/pacs_art_painting"
    # test_root = "datasets/pacs/art_painting"

    resnet18 = resnet18(num_classes=num_classes)
    m0 = get_cresnet(num_classes=num_classes, layers=[1,1,1])
    m1 = get_cresnet(num_classes=num_classes, layers=[2,2,1,1])
    m2 = get_cvit(num_classes=num_classes, image_size=image_size,patch_size=16)

    # for model, m_name in [(m1, "cresnet"), (m2, "cvit")]:
    # for model, cp_name in [(m0, f"cresnet_s_{d_name}")]:
    # for model, cp_name in [(m1, f"cresnet_l_{d_name}")]:
    for model, cp_name in [(resnet18, f"resnet18_{d_name}")]:
        train_tfm = get_inference_transform(image_size)
        train_ds = datasets.ImageFolder(root, transform=train_tfm)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        test_tfm = get_inference_transform(image_size)
        full_ds = datasets.ImageFolder(test_root, transform=test_tfm)
        sample_size = min(200, len(full_ds))
        indices = random.sample(range(len(full_ds)), sample_size)
        train_indices = indices[:180]
        val_indices = indices[180:]

        train_ds = datasets.ImageFolder(test_root, transform=train_tfm)
        val_ds = datasets.ImageFolder(test_root, transform=test_tfm)
        train_loader = DataLoader(
            Subset(train_ds, train_indices),
            batch_size=16,
            shuffle=True,
        )
        test_loader = DataLoader(
            Subset(val_ds, val_indices),
            batch_size=16,
            shuffle=False,
        )



        trainer = Trainer(
            model,
            loss_fn=nn.CrossEntropyLoss(),
            config=TrainConfig(
                epochs=200,
                accumulate_steps=1,
                early_stop=True,
                cp_name=f"{cp_name}.pth",
            ),
        )
        history = trainer.fit(train_loader, test_loader)
        trainer.save()

        # x,w1,w2,s=trainer.inference(test_loader)
        # print(s.shape)
