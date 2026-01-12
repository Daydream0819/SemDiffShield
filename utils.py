import torch
import torch.nn.functional as F
import os
import numpy as np


# ============================
# Image normalization
# ============================
# 当前训练流程：
# - Dataset 使用 transforms.ToTensor() → 图像范围 [0, 1]
# - Deep-JSCC 在 [0, 1] 空间训练
# 因此这里不再做任何缩放，保持恒等映射
def image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        return tensor
    return _inner


# ============================
# PSNR computation
# ============================
# 默认 max_val = 1.0，适配 [0,1] 图像空间
def get_psnr(image, gt, max_val=1.0, mse=None):
    if mse is None:
        mse = F.mse_loss(image, gt)

    if not isinstance(mse, torch.Tensor):
        mse = torch.tensor(mse)

    # 防止数值不稳定
    eps = 1e-10
    psnr = 10 * torch.log10((max_val ** 2) / (mse + eps))
    return psnr


# ============================
# Model saving
# ============================
def save_model(model, dir, path):
    os.makedirs(dir, exist_ok=True)
    flag = 1
    final_path = path

    while os.path.exists(final_path):
        final_path = path + '_' + str(flag)
        flag += 1

    torch.save(model.state_dict(), final_path)
    print(f"Model saved in {final_path}")


# ============================
# Reproducibility
# ============================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================
# Count model parameters
# ============================
def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param
