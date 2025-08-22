import torch
from torchvision import models
from atom.modules.vector_retrieval.data.dataset import CachedTensorDataset
from atom.modules.vector_retrieval.training.modules import VideoProcessor, Trainer

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=str, default="minkowski")
    parser.add_argument("--loss_func", type=str, default="triplet")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model = models.resnet18(weights="DEFAULT")
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.layer2.requires_grad_(True)
    model.layer3.requires_grad_(True)
    model.layer4.requires_grad_(True)
    processor = VideoProcessor(model=model, distance=args.distance, loss_func=args.loss_func, dtype=torch.float32)
    train_dataset = CachedTensorDataset("/mnt/bn/vector2/tensors_320x180_shots", end=3000, cache_size=100000, try_load_all=True)
    eval_dataset = CachedTensorDataset("/mnt/bn/vector2/tensors_320x180_shots", start=3000, cache_size=100000, try_load_all=True)
    optimizer = torch.optim.Adam(processor.parameters(), lr=1e-4)

    trainer = Trainer(
        model=processor, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        optimizer=optimizer, 
        train_batchsize=2560, 
        eval_batchsize=2560, 
        epoch=2,
        save_dir=f"/mnt/bn/vector2/ckpt-{args.loss_func}-{args.distance}-shots",
        save_interval=50,
        max_norm=2.0,
        shuffle=False,
    )
    trainer.train()
