import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from driftguard.model.c_resnet.model import get_cresnet
from driftguard.model.c_vit.model import get_cvit
from driftguard.model.dataset import get_inference_transform, get_train_transform
from driftguard.model.training.trainer import Trainer, TrainConfig
from torch.utils.data import Subset
import torch.nn as nn

for root, test_root, d_name in[
    ("datasets/init_subsets/pacs_art_painting", "datasets/pacs/art_painting", "pacs"),
    ("datasets/init_subsets/domainnet_clipart", "datasets/drift_domain_net/clipart", "ddn")
]:
    # root = "datasets/init_subsets/pacs_art_painting"
    # test_root = "datasets/pacs/art_painting"

    m1 = get_cresnet(num_classes=7, layers=[2,2,1,1])
    m2 = get_cvit(num_classes=7, image_size=224,patch_size=16)

    for model, m_name in [(m1, "cresnet"), (m2, "cvit")]:
        train_tfm = get_inference_transform(224)
        train_ds = datasets.ImageFolder(root, transform=train_tfm)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        test_tfm = get_inference_transform(224)
        test_ds = datasets.ImageFolder(test_root, transform=test_tfm)
        indices = random.sample(range(len(test_ds)), 100)
        test_ds = Subset(test_ds, indices)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)



        trainer = Trainer(
            model,
            loss_fn=nn.CrossEntropyLoss(),
            config=TrainConfig(
                epochs=200,
                accumulate_steps=1,
                early_stop=True,
                cp_name=f"{m_name}_{d_name}",
            ),
        )
        history = trainer.fit(test_loader,train_loader)
        trainer.save()

        # x,w1,w2,s=trainer.inference(test_loader)
        # print(s.shape)

