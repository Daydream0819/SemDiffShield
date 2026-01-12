import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL

# ======================
# 配置
# ======================
IMG_ROOT = "./data/custom512/train/class0"
OUT_ROOT = "./latent/custom512/train"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
BATCH_SIZE = 8
DEVICE = "cuda"

os.makedirs(OUT_ROOT, exist_ok=True)

# ======================
# 加载 VAE
# ======================
vae = AutoencoderKL.from_pretrained(
    MODEL_ID,
    subfolder="vae",
    torch_dtype=torch.float16
).to(DEVICE)
vae.eval()

# ======================
# 图像预处理
# ======================
tfm = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),          # [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3)  # → [-1,1]，VAE 要求
])

# ======================
# 读取图像
# ======================
img_files = sorted([
    os.path.join(IMG_ROOT, f)
    for f in os.listdir(IMG_ROOT)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

# ======================
# 编码
# ======================
idx = 0
with torch.no_grad():
    for i in tqdm(range(0, len(img_files), BATCH_SIZE)):
        batch_files = img_files[i:i+BATCH_SIZE]

        imgs = []
        for f in batch_files:
            img = Image.open(f).convert("RGB")
            imgs.append(tfm(img))
        imgs = torch.stack(imgs).to(DEVICE, dtype=torch.float16)

        # VAE encode
        latents = vae.encode(imgs).latent_dist.sample()
        latents = latents * 0.18215   # ★ 非常重要：SD 标准尺度

        for j in range(latents.shape[0]):
            torch.save(latents[j].cpu(), os.path.join(OUT_ROOT, f"{idx:06d}.pt"))
            idx += 1

print("Saved latents:", idx)
