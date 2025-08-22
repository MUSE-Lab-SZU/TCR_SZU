import torch
from torchvision import models
from atom.modules.vector_retrieval.training.utils import vit
from atom.modules.vector_retrieval.data.dataset import CachedTensorDataset
from atom.modules.vector_retrieval.training.modules import VideoProcessor, Trainer

if __name__ == '__main__':
    model = models.resnet18()
    model.fc = torch.nn.Identity()
    processor = VideoProcessor(model=model, distance="cosine", loss_func="triplet")
    ckpt = torch.load("/mnt/bn/vector-lf/ckpt-triplet-minkowski-layer234/epoch_1_step_119.pt", map_location="cpu")
    processor.load_state_dict(ckpt["model"])
    processor = processor.to(dtype=torch.float16)
    eval_dataset = CachedTensorDataset("/mnt/bn/vector2/tensors_320x180", start=3000, cache_size=100000, try_load_all=True)
    trainer = Trainer(
        model=processor, 
        eval_dataset=eval_dataset, 
        eval_batchsize=4000, 
        shuffle=False,
    )
    trainer.evaluate()
