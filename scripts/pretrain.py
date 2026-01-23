from collections import Counter
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from driftguard.exp import DATASET, MODEL
from driftguard.model.c_resnet.model import get_cresnet
from driftguard.model.c_vit.model import get_cvit
from driftguard.model.dataset import get_inference_transform, get_train_transform
from driftguard.model.training.trainer import Trainer, TrainConfig
from torch.utils.data import Subset
import torch.nn as nn
from torchvision.models import resnet18
import json
from pathlib import Path

for ds in [DATASET.DG5, DATASET.PACS, DATASET.DDN]:
    path, n_class, img_size = ds.value
    domain = json.loads(Path(path).read_text())["domains"][0]
    data_path = Path(path).parent / domain

    train_tfm, val_tfm = get_train_transform(img_size), get_inference_transform(img_size)

    full_ds = datasets.ImageFolder(data_path, transform=val_tfm)

    idxs = random.sample(range(len(full_ds)), 300)
    train_idxs_1, train_idxs_2, val_idxs  = (
        idxs[:100], idxs[100:200], idxs[200:300]
    )

    train_ds, val_ds = (
        datasets.ImageFolder(data_path, transform=train_tfm), 
        datasets.ImageFolder(data_path, transform=val_tfm)
    )

    train_loader_11, train_loader_12, train_loader_21, train_loader_22, test_loader = (
        DataLoader(Subset(val_ds, train_idxs_1), batch_size=16, shuffle=True),
        DataLoader(Subset(train_ds, train_idxs_1), batch_size=16, shuffle=True),
        DataLoader(Subset(val_ds, train_idxs_2), batch_size=16, shuffle=True),
        DataLoader(Subset(train_ds, train_idxs_2), batch_size=16, shuffle=True),
        DataLoader(Subset(val_ds, val_idxs), batch_size=16, shuffle=False),
    )
    # print("train_1:", Counter(train_ds_1.targets[i] for i in train_idxs_1))
    # print("train_2:", Counter(train_ds_2.targets[i] for i in train_idxs_2))
    # break

    for model in [MODEL.CRST_S, MODEL.CRST_M, MODEL.CVIT]:
        
        if model == MODEL.CRST_S and (ds == DATASET.PACS or ds == DATASET.DDN):
            continue  # skip cresnet_s on pacs and ddn
        if (model == MODEL.CVIT or model == MODEL.CRST_M) and ds == DATASET.DG5:
            continue  # skip cvit on dg5

        m = model.fn(n_class)
        cp_name = f"{ds.name}-{model.value}.pth" # -> name of checkpoint file

        
        print(f" \n ***Training {model.value} on {ds.name} ({domain}), saving to {cp_name}***")
        trainer = Trainer(
            m,
            loss_fn=nn.CrossEntropyLoss(),
            config=TrainConfig(
                epochs=80,
                accumulate_steps=1,
                early_stop=True,
                cp_name=cp_name,
            ),
        )
        history = trainer.fit(train_loader_11, test_loader)
        history = trainer.fit(train_loader_12, test_loader)

        history = trainer.fit(train_loader_21, test_loader)
        history = trainer.fit(train_loader_22, test_loader)
        trainer.save()

        

