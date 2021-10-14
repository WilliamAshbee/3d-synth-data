from vit_pytorch import *
import torch
v = ViT3d(
    image_size = 32,
    patch_size = 4,
    num_classes = 2000,
    dim = 2048,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    channels=1
)

v = torch.nn.Sequential(
    v,
    torch.nn.Sigmoid()
)

img = torch.randn(100,1, 32, 32,32)

preds = v(img)# (1, 1000)

print(preds)
