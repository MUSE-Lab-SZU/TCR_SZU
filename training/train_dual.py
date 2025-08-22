import torch
from torchvision import models
from atom.modules.vector_retrieval.data.dataset import CachedTensorDataset
from atom.modules.vector_retrieval.training.modules import VideoProcessorDual, Trainer


if __name__ == "__main__":
    # ckpt = torch.load("atom/modules/vector_retrieval/ckpt-vit-b-16-dual/epoch_1_step_188.pt")
    raw_encoder = models.resnet18(weights="DEFAULT")
    raw_encoder.fc = torch.nn.Linear(512, 1024)
    render_encoder = models.resnet18(weights="DEFAULT")
    render_encoder.fc = torch.nn.Linear(512, 1024)
    processor = VideoProcessorDual(raw_encoder=raw_encoder, render_encoder=render_encoder, temperature=0.8)
    processor.requires_grad_(False)
    processor.raw_encoder.fc.requires_grad_(True)
    processor.render_encoder.fc.requires_grad_(True)
    # processor.load_state_dict(ckpt["model"])
    train_dataset = CachedTensorDataset("/mnt/bn/vector2/tensors_224x224", end=800, try_load_all=False)
    eval_dataset = CachedTensorDataset("/mnt/bn/vector2/tensors_224x224", start=800, try_load_all=False)
    optimizer = torch.optim.Adam(processor.parameters(), lr=5e-4)
    # optimizer.load_state_dict(ckpt["optimizer"])

    trainer = Trainer(
        model=processor, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        optimizer=optimizer, 
        train_batchsize=1024, 
        eval_batchsize=2560, 
        epoch=8,
        save_dir="atom/modules/vector_retrieval/ckpt-resnet18-dual",
        save_interval=50,
        max_norm=2.0,
        shuffle=False,
        hard_topk=0,
    )
    trainer.train()
