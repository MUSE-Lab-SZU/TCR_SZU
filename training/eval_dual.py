import torch
from torchvision import models
from atom.modules.vector_retrieval.data.dataset import CachedTensorDataset
from atom.modules.vector_retrieval.training.modules import VideoProcessorDual, Trainer

if __name__ == '__main__':
    raw_encoder = models.resnet18(weights="DEFAULT")
    raw_encoder.fc = torch.nn.Linear(512, 1024)
    render_encoder = models.resnet18(weights="DEFAULT")
    render_encoder.fc = torch.nn.Linear(512, 1024)
    processor = VideoProcessorDual(raw_encoder=raw_encoder, render_encoder=render_encoder, temperature=0.8)
    ckpt = torch.load("atom/modules/vector_retrieval/ckpt-resnet18-dual/epoch_8_step_94.pt", map_location="cpu")
    processor.load_state_dict(ckpt["model"])
    eval_dataset = CachedTensorDataset("/mnt/bn/vector2/tensors_224x224", start=800, cache_size=100000, try_load_all=False)
    trainer = Trainer(
        model=processor, 
        eval_dataset=eval_dataset, 
        eval_batchsize=2560, 
        shuffle=False,
    )
    trainer.evaluate()
