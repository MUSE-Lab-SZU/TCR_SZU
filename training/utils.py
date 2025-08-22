import torch
from torchvision.models import resnet18
from atom.modules.vector_retrieval.training.modules import ViT
from atom.modules.vector_retrieval.training.modules import VideoProcessor

def vit(
    height=320,
    width=180,
    patch_size=20,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
    dropout=0.0,
    attention_dropout=0.0,
    num_classes=0,
):
    return ViT(
        height=height,
        width=width,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        num_classes=num_classes,
    )

def dump_model(ckpt_path:str, dump_path:str="/mnt/bn/vector2/model.pth"):
    ckpt = torch.load(ckpt_path, map_location="cpu")["model"]
    model = resnet18()
    model.fc = torch.nn.Linear(512, 512, bias=False)
    processor = VideoProcessor(model=model)
    processor.load_state_dict(ckpt)
    torch.save(processor, dump_path)

if __name__ == "__main__":
    dump_model("/mnt/bn/vector-lf/ckpt-triplet-minkowski-full/epoch_1_step_119.pt", "/mnt/bn/vector-lf/ckpt-triplet-minkowski-full/model.pth")