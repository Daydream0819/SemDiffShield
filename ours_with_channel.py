import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

from channel_R import Channel
from model_R import DeepJSCC

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
MODEL_ID = "runwayml/stable-diffusion-v1-5"


# ================= 1. 核心数学工具 =================

def get_keys_from_seed(key, length, device):
    """ 生成空间置乱的索引 """
    g = torch.Generator(device="cpu").manual_seed(key)
    perm_idx = torch.randperm(length, generator=g).to(device)
    inv_perm_idx = torch.argsort(perm_idx)
    return perm_idx, inv_perm_idx


def spatial_permute(latents, key, inverse=False):
    """ 空间置乱: 打碎结构 """
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c, -1)
    perm_idx, inv_perm_idx = get_keys_from_seed(key, h * w, latents.device)

    if not inverse:
        latents = latents[:, :, perm_idx]
    else:
        latents = latents[:, :, inv_perm_idx]

    return latents.reshape(b, c, h, w)


def get_orthogonal_matrix(key, dim, device):
    gen = torch.Generator(device="cpu").manual_seed(key)
    H = torch.randn(dim, dim, generator=gen)
    Q, R = torch.linalg.qr(H)
    return Q.to(device).to(DTYPE)


def orthogonal_transform(latents, key, inverse=False):
    """ 通道旋转 """
    b, c, h, w = latents.shape
    latents_perm = latents.permute(0, 2, 3, 1)
    Q = get_orthogonal_matrix(key, c, latents.device)
    if not inverse:
        res = torch.matmul(latents_perm, Q.t())
    else:
        res = torch.matmul(latents_perm, Q)
    return res.permute(0, 3, 1, 2)


def get_secret_trajectory(key, num_steps):
    """ 生成秘密时间步 """
    np.random.seed(key)
    full_range = np.arange(0, 1000)
    chosen = np.random.choice(full_range[1:], size=num_steps - 1, replace=False)
    chosen = np.sort(chosen)[::-1]
    chosen = np.append(chosen, 0)
    return torch.from_numpy(chosen.copy()).long().to(DEVICE)


def get_step_bound_embeds(base_embeds, t_val, secret_key):
    """ SL-DSM 扰动 """
    step_seed = int(secret_key) + int(t_val.item()) * 777
    gen = torch.Generator(device="cpu").manual_seed(step_seed)
    noise = torch.randn(base_embeds.shape, generator=gen).to(DEVICE).to(DTYPE)
    scale = 8.0
    return base_embeds + scale * noise


def statistical_correction(latents, target_mean=0.0, target_std=1.0):
    curr_mean = latents.mean()
    curr_std = latents.std()
    latents = (latents - curr_mean) / (curr_std + 1e-12)
    latents = latents * target_std + target_mean
    return latents, curr_mean, curr_std


def pil_to_tensor_01(img_pil: Image.Image) -> torch.Tensor:
    """PIL -> torch tensor in [0,1], shape [1,3,H,W], float32, on CPU"""
    img = img_pil.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return ten


@torch.no_grad()
def compute_mse_psnr(x: torch.Tensor, y: torch.Tensor):
    """x,y: [1,3,H,W] in [0,1]"""
    mse = torch.mean((x - y) ** 2).item()
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
    return mse, psnr


def try_init_ssim():
    """Return a callable ssim_fn(x,y)->float or None."""
    try:
        from pytorch_msssim import ssim as ssim_fn  # expects [0,1]

        def _ssim(x, y):
            return float(ssim_fn(x, y, data_range=1.0, size_average=True).item())
        return _ssim
    except Exception:
        return None


def try_init_lpips(device):
    """Return lpips_model(x,y)->float or None. Expects [-1,1]."""
    try:
        import lpips
        model = lpips.LPIPS(net="alex").to(device)
        model.eval()

        @torch.no_grad()
        def _lpips(x01, y01):
            x = x01.to(device) * 2 - 1
            y = y01.to(device) * 2 - 1
            v = model(x, y)
            return float(v.mean().item())
        return _lpips
    except Exception:
        return None


def try_init_clip(device):
    """
    Return clip_sim(x01,y01)->float or None.
    Uses transformers CLIPModel/CLIPProcessor.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        model_id = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id).to(device)
        proc = CLIPProcessor.from_pretrained(model_id)
        model.eval()

        @torch.no_grad()
        def _clip_sim(x_pil: Image.Image, y_pil: Image.Image):
            inputs = proc(images=[x_pil.convert("RGB"), y_pil.convert("RGB")],
                          return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_image_features(**inputs)  # [2, D]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sim = torch.sum(feats[0] * feats[1]).item()
            return float(sim)

        return _clip_sim
    except Exception:
        return None


def plot_curve(xs, ys, title, xlabel, ylabel, save_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def parse_snr_list(s: str):
    # e.g. "3,5,7,9,11"
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [float(p) for p in parts]


# ================= 2. Pipeline =================

class CryptoPipeline:
    def __init__(self, num_steps=150):
        self.scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=self.scheduler, safety_checker=None, torch_dtype=DTYPE
        ).to(DEVICE)
        self.pipe.unet.to(DTYPE)

        self.scheduler.set_timesteps(num_steps)
        self.num_steps = num_steps
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(DEVICE).to(DTYPE)
        self.null_embeds = self.encode_text("")

    def encode_text(self, prompt):
        text_input = self.pipe.tokenizer(
            [prompt], padding="max_length", max_length=self.pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            return self.pipe.text_encoder(text_input.input_ids.to(DEVICE))[0].to(DTYPE)

    @torch.no_grad()
    def run_ddim_loop(self, latents, trajectory, prompt_embeds, key=None, is_inversion=False, use_sl_dsm=False,
                      guidance_scale=1.0, desc=""):
        if is_inversion:
            timesteps = torch.sort(trajectory)[0]
        else:
            timesteps = torch.sort(trajectory, descending=True)[0]

        curr_latents = latents.clone()

        if guidance_scale > 1.0:
            uncond_embeds = self.null_embeds.repeat(latents.shape[0], 1, 1)
            text_embeds = prompt_embeds.repeat(latents.shape[0], 1, 1)

        for i in tqdm(range(len(timesteps) - 1), desc=f"{desc} (CFG={guidance_scale})", leave=False):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            current_cond = prompt_embeds
            if use_sl_dsm and key is not None:
                current_cond = get_step_bound_embeds(prompt_embeds, t_curr, key)
                if guidance_scale > 1.0:
                    text_embeds = current_cond

            if guidance_scale > 1.0:
                latent_input = torch.cat([curr_latents] * 2)
                combined_embeds = torch.cat([uncond_embeds, text_embeds])
                noise_pred_combined = self.pipe.unet(latent_input, t_curr, encoder_hidden_states=combined_embeds).sample
                noise_pred_uncond, noise_pred_text = noise_pred_combined.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = self.pipe.unet(curr_latents, t_curr, encoder_hidden_states=current_cond).sample

            alpha_curr = self.alphas_cumprod[t_curr]
            alpha_next = self.alphas_cumprod[t_next]
            beta_curr = 1 - alpha_curr
            beta_next = 1 - alpha_next

            pred_x0 = (curr_latents - beta_curr ** 0.5 * noise_pred) / alpha_curr ** 0.5

            curr_latents = alpha_next ** 0.5 * pred_x0 + beta_next ** 0.5 * noise_pred

        return curr_latents

    def image2latent(self, img_path):
        if not os.path.exists(img_path):
            print(f"⚠️ 图片不存在，生成灰图: {img_path}")
            Image.new('RGB', (512, 512), (128, 128, 128)).save(img_path)

        img = Image.open(img_path).convert("RGB").resize((512, 512))
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE).to(DTYPE)

        with torch.no_grad():
            latents = self.pipe.vae.encode(img).latent_dist.mean * self.pipe.vae.config.scaling_factor
        return latents

    def latent2image(self, latents, save_path):
        with torch.no_grad():
            latents = latents / self.pipe.vae.config.scaling_factor
            image = self.pipe.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).detach().numpy()[0]
            Image.fromarray((image * 255).astype(np.uint8)).save(save_path)


# ================= 3. 主程序 =================

def main():
    parser = argparse.ArgumentParser(description="运行隐写加密/解密流程 (with JSCC latent)")

    parser.add_argument("--steps", type=int, default=150, help="采样步数")
    parser.add_argument("--key", type=int, default=2024, help="加密密钥")
    parser.add_argument("--stego_cfg", type=float, default=3.0, help="伪装生成的引导系数")

    # ===== JSCC args =====
    parser.add_argument("--channel", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "RAYLEIGH"],
                        help="JSCC channel type")
    parser.add_argument("--jscc_snr", type=float, default=15.0, help="JSCC training SNR (used if not sweeping)")
    parser.add_argument("--jscc_ckpt", type=str, default="./epoch_450_A.pkl",
                        help="JSCC ckpt path")
    parser.add_argument("--jscc_c", type=int, default=32, help="(legacy) just for printing")

    # main I/O
    parser.add_argument("--image_path", type=str, default="example/data1/11068.png", help="原始图片路径")
    parser.add_argument("--public_emb", type=str, default=" a face of a young woman", help="公开的伪装提示词")

    # sweep
    parser.add_argument("--snr_list", type=str, default="0,2,4,6,8,10,12,14",
                        help="comma-separated SNR list for sweep")
    parser.add_argument("--sweep", default= True, action="store_true", help="启用 SNR sweep 评测并画曲线")
    parser.add_argument("--out_dir", type=str, default="./snr_eval", help="输出目录")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    if not os.path.exists(args.image_path):
        print(f"❌ 错误: 找不到图片文件 -> {args.image_path}")
        sys.exit(1)

    print(">>> 初始化 Pipeline...")
    pipeline = CryptoPipeline(num_steps=args.steps)

    ssim_fn = try_init_ssim()
    lpips_fn = try_init_lpips(device)
    clip_fn = try_init_clip(device)

    if ssim_fn is None:
        print("⚠️ SSIM 依赖未就绪（建议 pip install pytorch-msssim），将跳过 SSIM")
    if lpips_fn is None:
        print("⚠️ LPIPS 依赖未就绪（建议 pip install lpips），将跳过 LPIPS")
    if clip_fn is None:
        print("⚠️ CLIP 依赖未就绪（建议 pip install transformers 并确保可下载权重），将跳过 CLIP")

    print(f">>> 正在编码 Prompt: '{args.public_emb}'")
    secret_embeds = pipeline.encode_text("")
    public_embeds = pipeline.encode_text(args.public_emb)

    # ===== Load JSCC ONCE =====


    # input image & GT
    z0 = pipeline.image2latent(args.image_path)
    gt_pil = Image.open(args.image_path).convert("RGB").resize((512, 512))
    gt_01 = pil_to_tensor_01(gt_pil)

    # ========================================================
    # [Sender] 加密流程
    # ========================================================
    print("\n>>> [Sender] 加密流程...")

    standard_traj = pipeline.scheduler.timesteps
    secret_traj = get_secret_trajectory(args.key, args.steps)

    zT_encrypted = pipeline.run_ddim_loop(
        z0, secret_traj, secret_embeds,
        key=args.key, is_inversion=True, use_sl_dsm=True,
        guidance_scale=1.0, desc="1.加密反演"
    )

    z_perm = spatial_permute(zT_encrypted, args.key, inverse=False)
    z_enc = orthogonal_transform(z_perm, args.key, inverse=False)

    z_enc_norm, old_mean, old_std = statistical_correction(z_enc)

    z_stego = pipeline.run_ddim_loop(
        z_enc_norm, standard_traj, public_embeds,
        is_inversion=False, use_sl_dsm=False,
        guidance_scale=args.stego_cfg, desc="3.伪装生成"
    )

    stego_path = os.path.join(args.out_dir, "stego_final.png")
    pipeline.latent2image(z_stego, stego_path)
    print(f"✅ [Sender] 伪装图已保存: {stego_path}")

    # ========================================================
    # Sweep
    # ========================================================
    if args.sweep:
        snrs = parse_snr_list(args.snr_list)

        rows = []
        mse_list, psnr_list, ssim_list, lpips_list, clip_list = [], [], [], [], []

        print("\n>>> [Sweep] 开始逐 SNR 评测（仅正确 Key）...")

        for snr in snrs:
            print(f"\n===== SNR={snr} dB =====")

                # IMPORTANT: make JSCC channel use current snr
            jscc = DeepJSCC(
                channel_type = args.channel,
                hidden = 128,
                channel_ch=16,  # 选择合适的通道数，和潜在空间一致
                snr=snr,
                rx_ant=1
            ).to(device)
        
            jscc.load_state_dict(torch.load(args.jscc_ckpt, map_location=device))
            jscc.eval()
            with torch.no_grad():
                z_rx = jscc(z_stego.to(device))

            jscc_mse = torch.mean((z_rx - z_stego.to(device)) ** 2).item()
            print(f"📡 JSCC latent MSE on stego-latent @ {snr}dB = {jscc_mse:.6f}")

            # Receiver（正确 key）
            hat_z_enc_norm = pipeline.run_ddim_loop(
                z_rx, standard_traj, public_embeds,
                is_inversion=True, use_sl_dsm=False,
                guidance_scale=args.stego_cfg, desc="1.标准反演"
            )

            hat_z_enc = hat_z_enc_norm * old_std + old_mean
            hat_z_perm = orthogonal_transform(hat_z_enc, args.key, inverse=True)
            hat_zT = spatial_permute(hat_z_perm, args.key, inverse=True)

            secret_traj_rec = get_secret_trajectory(args.key, args.steps)
            hat_z0 = pipeline.run_ddim_loop(
                hat_zT, secret_traj_rec, secret_embeds,
                key=args.key, is_inversion=False, use_sl_dsm=True,
                guidance_scale=1.0, desc="4.加密恢复"
            )

            rec_path = os.path.join(args.out_dir, f"rec_final_snr{int(snr):02d}.png")
            pipeline.latent2image(hat_z0, rec_path)
            print(f"✅ 输出已保存: {rec_path}")

            rec_pil = Image.open(rec_path).convert("RGB").resize((512, 512))
            rec_01 = pil_to_tensor_01(rec_pil)

            mse, psnr = compute_mse_psnr(gt_01, rec_01)
            ssim = ssim_fn(gt_01, rec_01) if ssim_fn is not None else None
            lpips_v = lpips_fn(gt_01, rec_01) if lpips_fn is not None else None
            clip_v = clip_fn(gt_pil, rec_pil) if clip_fn is not None else None

            print(f"📊 Pixel MSE={mse:.6f} | PSNR={psnr:.3f} | "
                  f"SSIM={ssim if ssim is not None else 'NA'} | "
                  f"LPIPS={lpips_v if lpips_v is not None else 'NA'} | "
                  f"CLIP={clip_v if clip_v is not None else 'NA'}")

            rows.append({
                "snr": snr,
                "jscc_latent_mse": jscc_mse,
                "pixel_mse": mse,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_v,
                "clip_sim": clip_v,
                "rec_path": rec_path
            })

            mse_list.append(mse)
            psnr_list.append(psnr)
            ssim_list.append(ssim if ssim is not None else float("nan"))
            lpips_list.append(lpips_v if lpips_v is not None else float("nan"))
            clip_list.append(clip_v if clip_v is not None else float("nan"))

        # write csv
        csv_path = os.path.join(args.out_dir, "snr_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["snr", "jscc_latent_mse", "pixel_mse", "psnr", "ssim", "lpips", "clip_sim", "rec_path"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n✅ 指标已保存: {csv_path}")

        # plot curves
        plot_curve(snrs, mse_list, "Pixel MSE vs SNR", "SNR (dB)", "MSE",
                   os.path.join(args.out_dir, "curve_mse.png"))
        plot_curve(snrs, psnr_list, "PSNR vs SNR", "SNR (dB)", "PSNR (dB)",
                   os.path.join(args.out_dir, "curve_psnr.png"))

        if not np.all(np.isnan(np.array(ssim_list))):
            plot_curve(snrs, ssim_list, "SSIM vs SNR", "SNR (dB)", "SSIM",
                       os.path.join(args.out_dir, "curve_ssim.png"))
        if not np.all(np.isnan(np.array(lpips_list))):
            plot_curve(snrs, lpips_list, "LPIPS vs SNR", "SNR (dB)", "LPIPS",
                       os.path.join(args.out_dir, "curve_lpips.png"))
        if not np.all(np.isnan(np.array(clip_list))):
            plot_curve(snrs, clip_list, "CLIP Similarity vs SNR", "SNR (dB)", "Cosine Similarity",
                       os.path.join(args.out_dir, "curve_clip.png"))

        print(f"✅ 曲线图已保存到目录: {args.out_dir}")
        return


if __name__ == "__main__":
    main()
