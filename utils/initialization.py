import torch
import torch.nn.functional as F


def generate_edge_mask(img, threshold=None):
    gray = img.mean(dim=1, keepdim=True)
    kernel_x = img.new_tensor(((1, 0, -1), (2, 0, -2), (1, 0, -1))).view(1, 1, 3, 3)
    kernel_y = img.new_tensor(((1, 2, 1), (0, 0, 0), (-1, -2, -1))).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    grad = torch.sqrt(grad_x.square() + grad_y.square() + 1e-5)
    if threshold is None:
        threshold = grad.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
    return (grad > threshold).to(img.dtype), grad


class Init_interp:
    polar_offsets = {
        "90": (0, 0),
        "45": (0, 1),
        "135": (1, 0),
        "0": (1, 1),
    }
    angle_order = ["0", "45", "90", "135"]
    # Original color_polar_mosaic uses cv2-style BGR arrays.
    bgr_bayer_channels = {
        "BGGR": [0, 1, 1, 2],
        "RGGB": [2, 1, 1, 0],
        "GBRG": [1, 0, 2, 1],
        "GRBG": [1, 2, 0, 1],
    }
    # PIL/torchvision tensors are RGB. Real raw CPFA decoding should use this.
    rgb_bayer_channels = {
        "BGGR": [2, 1, 1, 0],
        "RGGB": [0, 1, 1, 2],
        "GBRG": [1, 2, 0, 1],
        "GRBG": [1, 0, 2, 1],
    }

    def __init__(self, phase="train", color_bayer_pattern="RGGB"):
        self.phase = phase
        if color_bayer_pattern not in self.bgr_bayer_channels:
            raise ValueError(f"Invalid COLOR_BAYER_PATTERN: {color_bayer_pattern}")
        self.color_bayer_pattern = color_bayer_pattern
        self.color_order = self.bgr_bayer_channels[color_bayer_pattern]
        self.raw_color_order = self.rgb_bayer_channels[color_bayer_pattern]

    def __call__(self, img):
        if img.dim() != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(img.shape)}")

        phase = self.phase.lower()
        if phase == "raw":
            return self.cpfa_to_initial(img, color_order=self.raw_color_order)

        gt = self._pad_to_multiple(img, 4)
        raw = self.gt_to_cpfa(gt)
        img_lr = self.cpfa_to_initial(raw, color_order=self.color_order)
        gt_lr = F.interpolate(gt, size=img_lr.shape[-2:], mode="bilinear", align_corners=False)
        if phase == "train":
            return img_lr, gt_lr
        return img_lr

    @staticmethod
    def _pad_to_multiple(img, multiple):
        _, _, h, w = img.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
        return img

    def gt_to_cpfa(self, gt):
        if gt.shape[1] != 12:
            raise ValueError(f"Expected 12-channel GT tensor, got shape {tuple(gt.shape)}")

        gt = self._pad_to_multiple(gt, 4)
        raw = torch.zeros(gt.shape[0], 1, gt.shape[2], gt.shape[3], dtype=gt.dtype, device=gt.device)
        angle_tensors = {
            "0": gt[:, 0:3],
            "45": gt[:, 3:6],
            "90": gt[:, 6:9],
            "135": gt[:, 9:12],
        }

        for angle, angle_tensor in angle_tensors.items():
            pr, pc = self.polar_offsets[angle]
            self._write_angle_to_cpfa(raw, angle_tensor, pr, pc)
        return raw

    def cpfa_to_initial(self, raw, color_order=None):
        if raw.shape[1] != 1:
            raw = raw.mean(dim=1, keepdim=True)
        raw = self._pad_to_multiple(raw, 4)
        color_order = self.color_order if color_order is None else color_order

        planes = []
        for angle in self.angle_order:
            pr, pc = self.polar_offsets[angle]
            sparse, mask = self._read_angle_from_cpfa(raw, pr, pc, color_order)
            planes.append(self._fill_bayer_sparse(sparse, mask))
        return torch.cat(planes, dim=1)

    def _write_angle_to_cpfa(self, raw, angle_tensor, pr, pc):
        # Preserve the checkpoint's BGR mosaic convention for synthetic RGGB input.
        raw[:, 0, pr + 0::4, pc + 0::4] = angle_tensor[:, self.color_order[0], pr + 0::4, pc + 0::4]
        raw[:, 0, pr + 0::4, pc + 2::4] = angle_tensor[:, self.color_order[1], pr + 0::4, pc + 2::4]
        raw[:, 0, pr + 2::4, pc + 0::4] = angle_tensor[:, self.color_order[2], pr + 2::4, pc + 0::4]
        raw[:, 0, pr + 2::4, pc + 2::4] = angle_tensor[:, self.color_order[3], pr + 2::4, pc + 2::4]

    def _read_angle_from_cpfa(self, raw, pr, pc, color_order):
        b, _, h, w = raw.shape
        sparse = torch.zeros(b, 3, h // 2, w // 2, dtype=raw.dtype, device=raw.device)
        mask = torch.zeros_like(sparse)

        positions = [
            (0, 0, color_order[0]),
            (0, 1, color_order[1]),
            (1, 0, color_order[2]),
            (1, 1, color_order[3]),
        ]
        for hr, hc, channel in positions:
            sparse[:, channel, hr::2, hc::2] = raw[:, 0, pr + 2 * hr::4, pc + 2 * hc::4]
            mask[:, channel, hr::2, hc::2] = 1.0
        return sparse, mask

    @staticmethod
    def _fill_bayer_sparse(sparse, mask, iterations=4):
        filled = sparse
        kernel = torch.ones(
            sparse.shape[1],
            1,
            3,
            3,
            dtype=sparse.dtype,
            device=sparse.device,
        )
        for _ in range(iterations):
            value_sum = F.conv2d(filled * mask, kernel, padding=1, groups=sparse.shape[1])
            weight_sum = F.conv2d(mask, kernel, padding=1, groups=sparse.shape[1])
            interp = value_sum / (weight_sum + 1e-5)
            missing = (mask == 0).to(sparse.dtype)
            filled = filled * mask + interp * missing
            mask = torch.clamp(mask + missing * (weight_sum > 0).to(sparse.dtype), 0, 1)
        return filled.clamp(0, 1)
