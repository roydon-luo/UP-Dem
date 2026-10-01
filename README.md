# UP-Dem

Official code and pretrained model for **UP-Dem: Deep Unrolling Convolutional Sparse Coding for Color Polarization Image Demosaicking**, published online in *IEEE Transactions on Image Processing* (2026). [Paper](https://doi.org/10.1109/TIP.2026.3730838).

UP-Dem reconstructs four RGB polarization views (0°, 45°, 90°, and 135°) from a color polarization filter array (CPFA) measurement. The model has a color reconstruction stage followed by a polarization reconstruction stage, with seven iterative CDU blocks plus the terminal block in each stage.

## Installation

Use Python 3.10 or newer. Install a [PyTorch and torchvision build](https://pytorch.org/get-started/locally/) suitable for your CPU or CUDA version, then:

```bash
pip install -r requirements.txt
```

The released checkpoint is [`checkpoints/updem.pth`](checkpoints/updem.pth). It contains model parameters only, without optimizer state. SHA-256: `333ddd1ee8fe9bb0a65e0dc67f9d719fe45462aad2dec351fc7d3e68dff80831`. `torch.load` should only be used with files from trusted sources.

## Quick start

The repository includes one real raw CPFA image and one four-angle OPID scene. From the repository root, run:

```bash
python infer.py --raw-root examples/raw --skip-rlp --output-dir outputs/quickstart --device cuda
python infer.py --rlp-root examples/paired --skip-raw --output-dir outputs/quickstart --device cuda
```

The reconstructed images are saved in `outputs/quickstart/Raw/album/` and `outputs/quickstart/RLP/OPID071/`. The paired example also produces `outputs/quickstart/RLP/metrics.csv`. Use `--device cpu` if CUDA is unavailable.

## Inference

For a folder of single-channel raw CPFA images (`.png`, `.jpg`, `.tif`, or `.tiff`):

```bash
python infer.py --raw-root /path/to/raw --skip-rlp --output-dir outputs/inference --device cuda
```

For paired full-resolution RGB polarization ground truth, organize scenes as below, then run the synthetic CPFA test:

```bash
python infer.py --rlp-root /path/to/paired --skip-raw --output-dir outputs/inference --device cuda
```

```text
paired/
  scene_001/
    0.png
    45.png
    90.png
    135.png
```

Use `--device cpu` when CUDA is unavailable. Results are written under `outputs/inference/Raw/` or `outputs/inference/RLP/`. Each scene includes the four reconstructed angle images, Stokes-derived visualizations, and a `.mat` file when SciPy is installed. Input images should use the same 4×4 CPFA arrangement and Bayer pattern as the paper; the default is RGGB with polarization offsets specified in `utils/initialization.py`. Verify the camera's pixel order before applying this checkpoint to another sensor.

Run both inputs together with `python infer.py --rlp-root /path/to/paired --raw-root /path/to/raw --output-dir outputs/inference --device cuda`. The RLP output includes `metrics.csv` with PSNR and SSIM against the four-angle input images.

## Training

Training accepts one or more roots with scene folders in the paired layout above. It also accepts the original dataset layout: `gt_0`, `gt_45`, `gt_90`, `gt_135` folders with matching `<scene>_<angle>.png` names. For example:

```bash
python train.py --train-root /path/to/MQ --train-root /path/to/PIDSR --train-root /path/to/OPID --output-dir outputs/train --device cuda
```

Defaults are 100 epochs, batch size 24, 128×128 paired random crops, Adam at 1e-4, and seven iterative CDU blocks. The staged loss uses reconstruction MSE for epochs 1–10, adds polarization loss for epochs 11–31, and uses self-similarity plus polarization losses afterward. The latest training checkpoint is saved as `model.pth` after each epoch. A quick integration run is:

```bash
python train.py --train-root /path/to/paired --batch-size 1 --max-steps 1 --output-dir outputs/smoke --device cuda
```

Training from scratch depends on the dataset and augmentation used. The command above does not recreate the exact training set or experimental environment used for the released checkpoint. The available training code uses a noise scale of 0.04, whereas the paper reports 0.1; this discrepancy is being checked against the original experiment settings.

## Data

OPID is our outdoor polarization image dataset with 80 distinct scenes at 1024×1024 resolution. A division-of-time system with a rotating polarizer captures RGB images at 0°, 45°, 90°, and 135° for each scene. The scenes include vehicles, electric scooters, buildings, and wider city views at different distances and under morning, noon, and evening illumination. For each angle, 50 frames are averaged to reduce sensor noise while keeping the four-angle acquisition of a scene to approximately one minute. These four-angle images provide reference data for synthetic CPFA evaluation and can be arranged in the paired layout shown above.

The full source datasets and OPID collection are not included in this code repository; the paired OPID scene above is a small example. A public OPID download location will be added when available. The dataset loader supports the formats described above; public MQ and PIDSR datasets are third-party resources governed by their own terms.

## Citation

```bibtex
@article{luo2026updem,
  title   = {UP-Dem: Deep Unrolling Convolutional Sparse Coding for Color Polarization Image Demosaicking},
  author  = {Luo, Yidong and Wu, Caiyun and Li, Chenggong and Wang, Ping and Yuan, Xin and Yang, Kailun and Zhang, Junchao},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026},
  doi     = {10.1109/TIP.2026.3730838}
}
```

## License and acknowledgments

The project code is released under the [MIT License](LICENSE). The SFI components in `utils/SFI.py` were adapted from [Restormer](https://github.com/swz30/Restormer), and the SSIM implementation in `utils/losses.py` was adapted from [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim). See [third-party notices](THIRD_PARTY_NOTICES.md).
