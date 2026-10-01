"""Train UP-Dem with the staged losses used by the research code."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from arguments import args
from utils.net import UPDem
from dataloader.dataset import PairedPolarizationDataset
from utils.losses import SelfSimilarityLoss, polar_loss
from utils.initialization import Init_interp, generate_edge_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Train UP-Dem on paired polarization images")
    parser.add_argument("--train-root", action="append", required=True,
                        help="Parent directory with dataset/scene/{0,45,90,135}.png; repeat for multiple roots")
    parser.add_argument("--output-dir", default=args.train_path)
    parser.add_argument("--epochs", type=int, default=args.epochs)
    parser.add_argument("--batch-size", type=int, default=args.train_batch_size)
    parser.add_argument("--crop-size", type=int, default=args.img_size)
    parser.add_argument("--learning-rate", type=float, default=args.lr)
    parser.add_argument("--device", default=args.device)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, help="Limit optimizer steps for a quick integration check")
    return parser.parse_args()


def main():
    opts = parse_args()
    if opts.crop_size != 128:
        raise ValueError("The self-similarity loss uses 128x128 crops; set --crop-size 128")
    if opts.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    device = torch.device(opts.device)
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(opts.seed)

    dataset = PairedPolarizationDataset(opts.train_root, crop_size=opts.crop_size)
    loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                        num_workers=opts.num_workers, pin_memory=device.type == "cuda")
    model = UPDem(img_channels=args.num_input_channels, in_channels=args.kernel_channel,
                  out_channels=args.kernel_channel, kernel_size=args.kernel_size,
                  ista_iters=args.ista_iters, act="prelu").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[opts.epochs // 2, opts.epochs], gamma=0.1)
    mse = nn.MSELoss()
    similarity = SelfSimilarityLoss().to(device)
    init = Init_interp(phase="train")
    out_dir = Path(opts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if opts.resume:
        checkpoint = torch.load(opts.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            for _ in range(start_epoch):
                scheduler.step()
    print(f"device={device} scenes={len(dataset)} epochs={opts.epochs} start={start_epoch}", flush=True)
    total_steps = 0
    for epoch in range(start_epoch, opts.epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            mask, _ = generate_edge_mask(batch)
            low, mid_target = init(batch)
            _, grad = generate_edge_mask(mid_target)
            noisy = low + torch.randn_like(low) * 0.04 * grad
            mid, final = model(noisy)
            mse_loss = mse(final, batch) + mse(mid, mid_target)
            if epoch < 10:
                loss = 10 * mse_loss
            elif epoch <= 30:
                loss = 10 * mse_loss + polar_loss(mid, mid_target) + polar_loss(final, batch)
            else:
                loss = (100 * similarity(batch, final, mask)
                        + polar_loss(mid, mid_target)
                        + polar_loss(final, batch))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_steps += 1
            print(f"epoch={epoch+1} step={total_steps} loss={loss.item():.6f}", flush=True)
            if opts.max_steps and total_steps >= opts.max_steps:
                break
        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, out_dir / "model.pth")
        if opts.max_steps and total_steps >= opts.max_steps:
            break


if __name__ == "__main__":
    main()
