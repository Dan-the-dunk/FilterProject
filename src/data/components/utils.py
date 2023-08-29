
import torch


IMG_MEAN = [0.485, 0.456, 0.406] 
IMG_STD = [0.229, 0.224, 0.225]
def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch. Tensor:
# 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1). permute(3, 0, 1, 2)