import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

try:
    from scipy.io import savemat
except Exception:
    savemat = None

from arguments import args
from utils.net import UPDem
from dataloader.dataset import PairedPolarizationDataset, RawPolarizationDataset
from utils.initialization import Init_interp
from utils.losses import _ssim


def parse_args():
    parser = argparse.ArgumentParser(description="UPDem full inference for RLP and Raw")
    parser.add_argument("--checkpoint", default=args.checkpoint_path)
    parser.add_argument("--rlp-root", default=args.val_data_path)
    parser.add_argument("--raw-root", default=args.raw_data_path)
    parser.add_argument("--output-dir", default=args.test_path)
    parser.add_argument("--device", default=args.device)
    parser.add_argument("--batch-size", type=int, default=args.test_batch_size)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-rlp", action="store_true")
    parser.add_argument("--skip-raw", action="store_true")
    return parser.parse_args()


def build_model(device, checkpoint_path):
    model = UPDem(
        img_channels=args.num_input_channels,
        in_channels=args.kernel_channel,
        out_channels=args.kernel_channel,
        kernel_size=args.kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        ista_iters=args.ista_iters,
        act="prelu",
        act_value=None,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_model(model, batch, device, mode):
    original_hw = batch.shape[-2:]
    batch = batch.to(device, non_blocking=True)
    img_lr = Init_interp(phase="GT_test" if mode == "RLP" else "Raw")(batch)
    _, output = model(img_lr)
    return output[..., :original_hw[0], :original_hw[1]], img_lr


def compute_stokes(img_tensor):
    im0 = img_tensor[:, 0:3]
    im45 = img_tensor[:, 3:6]
    im90 = img_tensor[:, 6:9]
    im135 = img_tensor[:, 9:12]
    s0 = (im0 + im45 + im90 + im135) * 0.5
    s1 = im0 - im90
    s2 = im45 - im135
    dolp = torch.sqrt(s1.pow(2) + s2.pow(2)) / (s0 + 1e-5)
    aolp = (0.5 * torch.atan2(s2, s1) + torch.pi / 2) / torch.pi
    return s0, torch.clamp(dolp, 0, 1), torch.clamp(aolp, 0, 1)


def save_prediction(output, init, sample_dir, sample_name):
    sample_dir.mkdir(parents=True, exist_ok=True)
    output = output.detach().cpu().clamp(0, 1)
    init = init.detach().cpu().clamp(0, 1)

    names = ["0", "45", "90", "135"]
    for idx, name in enumerate(names):
        start = idx * 3
        save_image(output[start:start + 3], sample_dir / f"{name}.png")
        save_image(init[start:start + 3], sample_dir / f"ini_{name}.png")

    s0, dolp, aolp = compute_stokes(output.unsqueeze(0))
    ini_s0, ini_dolp, ini_aolp = compute_stokes(init.unsqueeze(0))
    save_image(s0[0], sample_dir / "S0.png")
    save_image(dolp[0], sample_dir / "DoLP.png")
    save_image(aolp[0], sample_dir / "AoLP.png")
    save_image(ini_s0[0], sample_dir / "ini_S0.png")
    save_image(ini_dolp[0], sample_dir / "ini_DoLP.png")
    save_image(ini_aolp[0], sample_dir / "ini_AoLP.png")

    if savemat is not None:
        mat_data = {
            "imOut_0": output[0:3].permute(1, 2, 0).numpy(),
            "imOut_45": output[3:6].permute(1, 2, 0).numpy(),
            "imOut_90": output[6:9].permute(1, 2, 0).numpy(),
            "imOut_135": output[9:12].permute(1, 2, 0).numpy(),
        }
        savemat(sample_dir / f"{sample_name}.mat", mat_data)


def infer_dataset(model, opts, device, mode):
    dataset = (PairedPolarizationDataset(opts.rlp_root, return_name=True) if mode == "RLP"
               else RawPolarizationDataset(opts.raw_root, return_name=True))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=device.type == "cuda",
    )
    out_root = Path(opts.output_dir) / mode
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{mode} images: {len(dataset)}")
    metrics = []
    with torch.inference_mode():
        for batch, names in loader:
            output, init = run_model(model, batch, device, mode)
            init_full = F.interpolate(init, size=output.shape[-2:], mode="bilinear", align_corners=False)
            for i, name in enumerate(names):
                save_prediction(output[i], init_full[i], out_root / name, Path(name).name)
                if mode == "RLP":
                    target = batch[i:i + 1].to(device)
                    pred = output[i:i + 1].clamp(0, 1)
                    mse = F.mse_loss(pred, target).item()
                    psnr = float("inf") if mse == 0 else -10 * torch.log10(torch.tensor(mse)).item()
                    ssim = _ssim(pred, target).item()
                    metrics.append((name, psnr, ssim))
                    print(f"{mode}/{name}: PSNR={psnr:.3f} dB SSIM={ssim:.5f}")
                else:
                    finite = bool(torch.isfinite(output[i]).all())
                    print(f"{mode}/{name}: shape={tuple(output[i].shape)} finite={finite}")
                    if not finite:
                        raise ValueError(f"Non-finite output for {name}")
    if metrics:
        with (out_root / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("scene", "psnr_db", "ssim"))
            writer.writerows(metrics)
        print(f"RLP mean: PSNR={sum(row[1] for row in metrics) / len(metrics):.3f} dB "
              f"SSIM={sum(row[2] for row in metrics) / len(metrics):.5f}")


def main():
    opts = parse_args()
    device = torch.device(opts.device if not opts.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {opts.checkpoint}")
    if savemat is None:
        print("scipy is not available; .mat files will be skipped.")
    model = build_model(device, opts.checkpoint)

    if not opts.skip_rlp:
        infer_dataset(model, opts, device, "RLP")
    if not opts.skip_raw:
        infer_dataset(model, opts, device, "Raw")
    print("Inference complete")


if __name__ == "__main__":
    main()
