import os
import cv2
import math
from tqdm import tqdm
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models.vision_transformer import Encoder

from atom.utils.logger import logger

class ViT(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # _log_api_usage_once(self)
        torch._assert(height % patch_size == 0, "Input height indivisible by patch size!")
        torch._assert(width % patch_size == 0, "Input width indivisible by patch size!")
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        seq_length = (height // patch_size) * (width // patch_size)
        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length
        if num_classes > 0:
            self.heads = nn.Linear(hidden_dim, num_classes, bias=False)
        else:
            self.heads = nn.Identity()

        # Init the  patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))

        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.height, f"Wrong image height! Expected {self.height} but got {h}!")
        torch._assert(w == self.width, f"Wrong image width! Expected {self.width} but got {w}!")
        n_h = h // p
        n_w = w // p
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.heads(x)
        return x

class Resize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

class ConvertColor:
    def __call__(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class VideoProcessor(nn.Module):
    def __init__(
        self, 
        model:nn.Module,
        height:int=320,
        width:int=180,
        distance:str="minkowski",
        loss_func:str="triplet",
        margin:float=1.0,
        temperature:float=1.0,
        device:str="cuda" if torch.cuda.is_available() else "cpu",
        dtype:torch.dtype=torch.float32,
    ):
        super().__init__()
        assert distance in ["cosine", "minkowski"], f"距离函数只支持cosine和minkowski，不支持{distance}"
        assert loss_func in ["info_nce", "triplet"], f"损失函数只支持info_nce和triplet，不支持{loss_func}"
        self.transform = transforms.Compose([
            ConvertColor(),
            Resize(height, width),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.distance = distance
        self.loss_func = loss_func
        self.margin = margin
        self.temperature = temperature
        self.device = device
        self.dtype = dtype
        self.model = model.to(device=device, dtype=dtype)

    def to(self, device=None, dtype=None, **kwargs):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.model = self.model.to(device=device, dtype=dtype, **kwargs)
        return self
    
    def extract_features(self, frames:torch.Tensor):
        # frame: [B, C, H, W]
        frames = frames.to(device=self.device, dtype=self.dtype)
        features = self.model(frames) # [B, D, 1, 1]
        features = features.squeeze(-1).squeeze(-1) # [B, D]
        features = F.normalize(features, dim=-1)
        return features

    def extract_frame(
        self, 
        video_path:str,
        framestamps:list,
    ):
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件{video_path}")
        for stamp in framestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, stamp)
            success, frame = cap.read()
            if success:
                frames.append(frame)
            else:
                logger.info(f"{os.path.basename(video_path)}无法读取帧{stamp}")
        cap.release()
        return frames
    
    def transform_frames(self, frames:list):
        return torch.cat([self.transform(f).unsqueeze(0) for f in frames], dim=0)
    
    def topk_acc(
        self, 
        similarity:torch.Tensor, 
        labels:torch.Tensor, 
        k:int=1, 
        largest:bool=True,
    ):
        batchsize = similarity.shape[0]
        topk = min(k, batchsize)
        topk_preds = similarity.topk(k=topk, dim=-1, largest=largest).indices
        correct_topk = topk_preds.eq(labels.unsqueeze(1)).any(dim=1).sum()
        acc = correct_topk / batchsize
        return round(acc.item()*100, 2)
    
    def info_nce(self, embeddings1, embeddings2, temperature):
        assert embeddings1.shape == embeddings2.shape, f"用于计算loss的两个表征形状不同:{embeddings1.shape} != {embeddings2.shape}"
        batchsize = embeddings1.shape[0]
        if self.distance == "cosine":
            similarity = torch.matmul(embeddings1, embeddings2.t()) / temperature
        elif self.distance == "minkowski":
            similarity = 1 - 0.5 * torch.cdist(embeddings1, embeddings2, p=2) / temperature
        labels = torch.arange(batchsize).to(self.device)
        loss = F.cross_entropy(similarity, labels, reduction="none")
        top1_acc = self.topk_acc(similarity, labels)
        top3_acc = self.topk_acc(similarity, labels, k=3)
        return {"loss": loss, "top1_acc": top1_acc, "top3_acc": top3_acc}

    def triplet(self, embeddings1, embeddings2, margin=1.0):
        """
        Args:
            embeddings1: Tensor [B, D] - anchors
            embeddings2: Tensor [B, D] - positives (aligned)
            margin: float - margin value for loss
        Returns:
            loss: scalar Tensor
        """
        B = embeddings1.size(0)
        # Pairwise Euclidean distances: [B, B]
        if self.distance == "minkowski":
            dists = torch.cdist(embeddings1, embeddings2, p=2)  # shape: [B, B]
        elif self.distance == "cosine":
            dists = 2 - 2 * torch.matmul(embeddings1, embeddings2.t())  # equal cdist numerically but not gradiently 
        # Mask out diagonal (positive pair itself, i == j)
        mask = ~torch.eye(B, dtype=torch.bool, device=embeddings1.device)
        # Get labels for positive pairs
        labels = torch.arange(B).to(self.device)
        # Positive distances are the diagonal
        pos_dists = torch.diag(dists).unsqueeze(1)  # shape: [B, 1]

        # Compute loss for all negatives in the batch (excluding positive)
        losses = F.relu(pos_dists - dists + margin)  # shape: [B, B]
        loss = losses[mask].view(B, B - 1).mean(dim=-1)  # shape: [B, 1]

        top1_acc = self.topk_acc(dists, labels, largest=False)
        top3_acc = self.topk_acc(dists, labels, k=3, largest=False)
        return {"loss": loss, "top1_acc": top1_acc, "top3_acc": top3_acc}
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_step(self, batch):
        raw = batch["raw"]
        render = batch["render"]
        embeddings = self.extract_features(torch.cat([raw, render]))
        batchsize = int(embeddings.shape[0] / 2)
        raw_embeddings = embeddings[:batchsize]
        render_embeddings = embeddings[batchsize:]
        if self.loss_func == "triplet":
            return self.triplet(render_embeddings, raw_embeddings, self.margin)
        elif self.loss_func == "info_nce":
            return self.info_nce(render_embeddings, raw_embeddings, self.temperature)

    def eval_step(self, batch):
        raw = batch["raw"]
        render = batch["render"]
        embeddings = self.extract_features(torch.cat([raw, render]))
        batchsize = int(embeddings.shape[0] / 2)
        raw_embeddings = embeddings[:batchsize]
        render_embeddings = embeddings[batchsize:]
        if self.loss_func == "info_nce":
            loss_dict = self.info_nce(render_embeddings, raw_embeddings, self.temperature)
            loss_dict["loss"] = loss_dict.pop("loss").mean().item()
        elif self.loss_func == "triplet":
            loss_dict = self.triplet(render_embeddings, raw_embeddings, self.margin)
            loss_dict["loss"] = loss_dict.pop("loss").mean().item()
        return loss_dict

# class VideoProcessorDual(nn.Module):
#     def __init__(
#         self, 
#         raw_encoder:nn.Module,
#         render_encoder:nn.Module,
#         height:int=224,
#         width:int=224,
#         temperature:float=1.0,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#     ):
#         super().__init__()  
#         self.transform = transforms.Compose([
#             Resize(height, width),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#         ])
#         self.temperature = temperature
#         self.device = device
#         self.raw_encoder = raw_encoder.to(device)
#         self.render_encoder = render_encoder.to(device)

#     def extract_features(self, frames:torch.Tensor, type:str="raw"):
#         # frame: [B, C, H, W]
#         frames = frames.to(self.device)
#         if type == "raw":
#             features = self.raw_encoder(frames) # [B, D, 1, 1]
#         elif type == "render":
#             features = self.render_encoder(frames) # [B, D, 1, 1]
#         else:
#             raise ValueError(f"type must be raw or render, but got {type}")
#         features = features.squeeze(-1).squeeze(-1) # [B, D]
#         features = F.normalize(features, dim=-1)
#         return features
    
#     def info_nce(self, embeddings1, embeddings2):
#         assert embeddings1.shape == embeddings2.shape, f"用于计算loss的两个表征形状不同:{embeddings1.shape} != {embeddings2.shape}"
#         batchsize = embeddings1.shape[0]
#         similarity = torch.matmul(embeddings1, embeddings2.t()) / self.temperature
#         labels = torch.arange(batchsize).to(self.device)
#         loss1 = F.cross_entropy(similarity, labels, reduction="none")
#         loss2 = F.cross_entropy(similarity.t(), labels, reduction="none")
#         loss = (loss1 + loss2) / 2
#         acc = torch.eq(similarity.argmax(dim=-1), labels).sum() / len(labels)
#         return {"loss": loss, "acc": round(acc.item()*100, 2)}

#     # def train_step(self, batch):
#     #     raw = batch["raw"]
#     #     render = batch["render"]
#     #     raw_embeddings = self.extract_features(raw, type="raw")
#     #     render_embeddings = self.extract_features(render, type="render")
#     #     cross_loss = self.info_nce(raw_embeddings, render_embeddings)
#     #     raw_loss = self.info_nce(raw_embeddings, raw_embeddings)
#     #     loss = (cross_loss["loss"] + raw_loss["loss"]) / 2
#     #     return {"loss": loss, "inner_loss": round(raw_loss["loss"].mean().item(), 4), "cross_loss": round(cross_loss["loss"].mean().item(), 4), "acc": cross_loss["acc"]}

#     # def train_step(self, batch):
#     #     raw = batch["raw"]
#     #     render = batch["render"]
#     #     raw_embeddings = self.extract_features(raw, type="raw")
#     #     render_embeddings = self.extract_features(render, type="render")
#     #     return self.info_nce(raw_embeddings, render_embeddings)

#     def train_step(self, batch):
#         raw = batch["raw"]
#         render = batch["render"]

#         raw_embeddings = self.extract_features(raw, type="raw")
#         render_embeddings = self.extract_features(render, type="render")

#         cross_loss = self.info_nce(raw_embeddings, render_embeddings)
#         render_loss = self.info_nce(render_embeddings, render_embeddings)
#         raw_loss = self.info_nce(raw_embeddings, raw_embeddings)
#         loss = (cross_loss["loss"] + raw_loss["loss"] + render_loss["loss"]) / 3
#         return {
#             "loss": loss, 
#             "cross_loss": round(cross_loss["loss"].mean().item(), 4), 
#             "raw_loss": round(raw_loss["loss"].mean().item(), 4), 
#             "render_loss": round(render_loss["loss"].mean().item(), 4),
#             "acc": cross_loss["acc"],
#         }
    
#     def eval_step(self, batch):
#         raw = batch["raw"]
#         render = batch["render"]
#         raw_embeddings = self.extract_features(raw, type="raw")
#         render_embeddings = self.extract_features(render, type="render")
#         info_nce = self.info_nce(raw_embeddings, render_embeddings)
#         info_nce["loss"] = round(info_nce["loss"].mean().item(), 4)
#         return info_nce

class Trainer:
    def __init__(
        self,
        model,
        train_dataset=None,
        eval_dataset=None,
        train_batchsize=16,
        eval_batchsize=16,
        save_dir="./checkpoint",
        save_interval=64,
        optimizer=None,
        epoch=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_norm=2.0,
        shuffle=True,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.train_batchsize = train_batchsize
        self.eval_batchsize = eval_batchsize
        self.epoch = epoch
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.device = device
        self.max_norm = max_norm
        self.shuffle = shuffle
        os.makedirs(self.save_dir, exist_ok=True)

    def clip_gradients(self):
        # 提取出带梯度的参数
        parameters = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        # 使用 PyTorch 提供的 clip_grad_norm_
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, norm_type=2)
        return total_norm.cpu()

    def train(self):
        self.model.train()
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.train_batchsize, shuffle=self.shuffle
        )
        for ep in range(self.epoch):
            epoch_loss = 0.0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for step, batch in pbar:
                self.optimizer.zero_grad()

                step_results = self.model.train_step(batch)
                loss = step_results.pop("loss")

                final_loss = loss.mean()
                final_loss.backward()

                grad_norm = self.clip_gradients()
                self.optimizer.step()

                epoch_loss += final_loss.item()
                progress = ep + (step+1) / len(train_loader)
                step_results = {"loss": round(final_loss.item(), 4), **step_results, "grad_norm": round(grad_norm.item(), 4), "epoch": round(progress, 4)}
                tqdm.write(str(step_results))

                if (ep*len(train_loader)+step+1) % self.save_interval == 0 or step == len(train_loader)-1:
                    save_path = os.path.join(self.save_dir, f"epoch_{ep+1}_step_{step+1}.pt")
                    self.save(save_path)
                    tqdm.write(f"Model saved at {save_path}")

            tqdm.write(f"[Epoch {ep+1}] Avg Loss: {epoch_loss / len(train_loader):.4f}")
            self.evaluate()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        eval_loader = DataLoader(
            self.eval_dataset, batch_size=self.eval_batchsize, shuffle=self.shuffle
        )
        record = defaultdict(list)
        for batch in tqdm(eval_loader, desc="Evaluating"):
            metrics = self.model.eval_step(batch)
            tqdm.write(str(metrics))
            for k in metrics:
                record[k].append(metrics[k])
        for k in record:
            record[k] = sum(record[k]) / len(record[k])
        tqdm.write(f"Avg Metrics:{dict(record)}")

    def save(self, save_path):
        ckpt = {"model": self.model.state_dict()}
        if self.optimizer is not None:
            ckpt["optimizer"] = self.optimizer.state_dict()
        torch.save(ckpt, save_path)
