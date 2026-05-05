import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FullFaceNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        # 1. The Pre-Trained Brain
        # We set pretrained=None here so your local server doesn't waste 
        # time downloading weights from the internet every time it boots. 
        # It will use the weights from your local .pth file instead.
        self.backbone = InceptionResnetV1(pretrained=None, classify=False)
        
        # 2. Optimized Embedding Head (Linear + LayerNorm)
        self.custom_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.LayerNorm(512), 
            nn.PReLU(),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        embeddings = self.custom_head(x)
        return embeddings