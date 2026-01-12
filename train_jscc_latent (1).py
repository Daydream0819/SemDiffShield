import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_latent import LatentDataset
from model_R import DeepJSCC  # 使用你上传的新的模型类
from channel_R import Channel 
device= "cuda" if torch.cuda.is_available() else "cpu"
class MixedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.alpha = alpha # 0.8 MSE + 0.2 L1

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.l1(pred, target)



def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_root", default="./latent/custom512/train", type=str)
    p.add_argument("--out_dir", default="./ckpt_latent", type=str)

    p.add_argument("--snr", default=15.0, type=float)  # 修改为SNR=15
    p.add_argument("--channel", default="AWGN", choices=["AWGN", "Rayleigh"])
    p.add_argument("--in_ch", default=4, type=int)
    p.add_argument("--out_ch", default=4, type=int)
    p.add_argument("--hidden", default=128, type=int)

    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--epochs", default=1000, type=int)  # 训练1000个epoch
    p.add_argument("--lr", default=2e-4, type=float)  # 设置较小的学习率
    p.add_argument("--weight_decay", default=0.0, type=float)
    p.add_argument("--num_workers", default=2, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda:0", type=str)

    # two-stage training
    p.add_argument("--no_channel", action="store_true", help="disable channel (snr=None) for sanity check")
    p.add_argument("--pretrain_epochs", default=0, type=int, help="pretrain without channel for N epochs, then enable channel")
    p.add_argument("--save_every", default=50, type=int)

    # stability
    p.add_argument("--grad_clip", default=1.0, type=float, help="0 means disable")
    p.add_argument("--amp", action="store_true", help="use torch autocast + GradScaler")

    # Scheme A: feed CSI (h) into refine network
    p.add_argument("--use_csi", action="store_true", help="concat (y,h) as input to refine network")

    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def dataset_stats(ds: LatentDataset, n=20):
    n = min(n, len(ds))
    means, stds, pows = [], [], []
    for i in range(n):
        z = ds[i]
        means.append(float(z.mean().item()))
        stds.append(float(z.std().item()))
        pows.append(float(z.pow(2).mean().item()))
    return float(np.mean(means)), float(np.mean(stds)), float(np.mean(pows))


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- data ----
    ds = LatentDataset(args.train_root)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    m, s, p = dataset_stats(ds, n=20)
    print(f"[DATA] N={len(ds)} sample_mean≈{m:.4f} sample_std≈{s:.4f} sample_E[x^2]≈{p:.4f}")

    # ---- model ----
    # IMPORTANT FIX:
    # if pretrain_epochs > 0, we must start with snr=None (channel off)
    snr_init = None if (args.no_channel or args.pretrain_epochs > 0) else args.snr

    # normalize channel type naming
    channel_type = args.channel


    model = DeepJSCC(
        channel_type = channel_type,
        hidden = 128,
        channel_ch=16,  # 选择合适的通道数，和潜在空间一致
        snr=snr_init,
        rx_ant=1
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    criterion = MixedLoss(alpha=0.1).to(device) # 甚至可以主要用 L1 (alpha=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_loss = float("inf")
    best_path = os.path.join(args.out_dir, "jscc_latent_best.pkl")

    print(f"[INFO] train_root={args.train_root}")
    print(f"[INFO] channel={args.channel} snr={args.snr}dB | in_ch={args.in_ch} out_ch={args.out_ch} hidden={args.hidden}")
    print(f"[INFO] device={device} | batch={args.batch_size} | epochs={args.epochs} | lr={args.lr} | amp={args.amp}")
    print(f"[INFO] use_csi={args.use_csi} (Scheme A: concat(y,h))")

    if args.no_channel:
        print("[MODE] no_channel=True -> pure identity/refine sanity check (channel off)")
    if args.pretrain_epochs > 0:
        print(f"[MODE] two-stage: pretrain {args.pretrain_epochs} epochs with snr=None, then finetune with snr={args.snr}")

    t0 = time.time()
    for epoch in range(args.epochs):
        # switch from pretrain(no channel) -> finetune(with channel)
        if args.pretrain_epochs > 0 and epoch == args.pretrain_epochs:
            model.change_channel(channel_type, args.snr)
            print(f"[SWITCH] epoch={epoch}: channel enabled, type={channel_type}, snr={args.snr}")

        model.train()
        run = 0.0

        for z in tqdm(dl, desc=f"Epoch {epoch}", leave=False):
            z = z.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                z_hat = model(z)
                loss = criterion(z_hat, z)

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            run += float(loss.detach().cpu().item())

        epoch_loss = run / len(dl)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"epoch_{epoch+1}.pkl"))

        print(f"Epoch {epoch:03d} | train_mse={epoch_loss:.6f} | best={best_loss:.6f}")
        scheduler.step()
        print(f"Epoch {epoch} LR: {scheduler.get_last_lr()[0]:.2e}")
    print(f"[DONE] best_train_mse={best_loss:.6f}")
    print(f"[CKPT] {best_path}")
    print(f"[TIME] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
