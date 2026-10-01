import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gradient(image):
    kernels = image.new_tensor((
        ((1, 0, -1), (2, 0, -2), (1, 0, -1)),
        ((1, 2, 1), (0, 0, 0), (-1, -2, -1)),
    ))
    channels = image.shape[1]
    return tuple(F.conv2d(image, kernel.reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1),
                          padding=1, groups=channels) for kernel in kernels)


def _gradient_loss(prediction, target):
    px, py = _gradient(prediction)
    tx, ty = _gradient(target)
    return 0.8 * (F.l1_loss(px, tx) + F.l1_loss(py, ty)) + 0.4 * F.l1_loss(prediction, target)


def _ssim(prediction, target, window_size=11):
    coordinate = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype) - window_size // 2
    gaussian = torch.exp(-coordinate.square() / (2 * 1.5 ** 2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).reshape(1, 1, window_size, window_size)
    channels = prediction.shape[1]
    window = window.repeat(channels, 1, 1, 1)
    conv = lambda image: F.conv2d(image, window, padding=window_size // 2, groups=channels)
    mu_pred = conv(prediction)
    mu_target = conv(target)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_pred * mu_target
    var_pred = conv(prediction.square()) - mu_pred_sq
    var_target = conv(target.square()) - mu_target_sq
    covariance = conv(prediction * target) - mu_product
    return (((2 * mu_product + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) /
            ((mu_pred_sq + mu_target_sq + 0.01 ** 2) * (var_pred + var_target + 0.03 ** 2))).mean()


def _adaptive_tv(image):
    dx = image[..., 1:] - image[..., :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    weight_x = torch.exp(-F.pad(dx.abs(), (0, 1)))
    weight_y = torch.exp(-F.pad(dy.abs(), (0, 0, 0, 1)))
    left = F.pad(image, (1, 0, 0, 0), mode="replicate")[..., :-1]
    above = F.pad(image, (0, 0, 1, 0), mode="replicate")[..., :-1, :]
    return (weight_x * (image - left).abs()).mean() + (weight_y * (image - above).abs()).mean()


def polar_loss(output, target):
    out_0, out_45, out_90, out_135 = output.chunk(4, dim=1)
    gt_0, gt_45, gt_90, gt_135 = target.chunk(4, dim=1)
    out_s0 = (out_0 + out_45 + out_90 + out_135) / 2
    out_s1, out_s2 = out_0 - out_90, out_45 - out_135
    gt_s0 = (gt_0 + gt_45 + gt_90 + gt_135) / 2
    gt_s1, gt_s2 = gt_0 - gt_90, gt_45 - gt_135
    epsilon = 1e-5
    out_aop = torch.atan2(out_s2 + epsilon, out_s1 + epsilon) / 2
    gt_aop = torch.atan2(gt_s2 + epsilon, gt_s1 + epsilon) / 2
    out_dop = (torch.sqrt(out_s1.square() + out_s2.square() + epsilon) / (out_s0 + epsilon)).clamp(0, 1)
    gt_dop = (torch.sqrt(gt_s1.square() + gt_s2.square() + epsilon) / (gt_s0 + epsilon)).clamp(0, 1)
    grad = (_gradient_loss(output, target) + _gradient_loss(out_s0, gt_s0)
            + 10 * _gradient_loss(out_s1, gt_s1) + 10 * _gradient_loss(out_s2, gt_s2))
    stokes = F.l1_loss(out_s1, gt_s1) + F.l1_loss(out_s2, gt_s2)
    angle = F.l1_loss(out_aop, gt_aop) + 10 * F.l1_loss(out_dop, gt_dop)
    polar = F.l1_loss(out_0 + out_90, out_45 + out_135)
    structure = (4 - _ssim(output, target) - _ssim(out_s0, gt_s0)
                 - _ssim(out_s1, gt_s1) - _ssim(out_s2, gt_s2))
    dop_structure = 1 - _ssim(out_dop, gt_dop)
    tv = _adaptive_tv(torch.sqrt(out_s1.square() + out_s2.square() + epsilon))
    return grad + 100 * stokes + 10 * angle + 10 * polar + 20 * dop_structure + 10 * structure + 1000 * tv


def _similarity(image, mask):
    batch, channels, _, _ = image.shape
    search_size, window_size = 5, 9
    search = F.unfold(F.pad(image, (2, 2, 2, 2), mode="reflect"), kernel_size=search_size)
    index = torch.where(mask.flatten() == 1)[0]
    count = index.numel()
    search = search[:, :, index].reshape(batch, channels, search_size ** 2, count)
    search = search.permute(0, 1, 3, 2).reshape(batch, channels * count, search_size, search_size)
    search = F.unfold(search, kernel_size=window_size, padding=window_size // 2)
    search = search.reshape(batch, channels, count, window_size ** 2, search_size ** 2)
    search = search.permute(0, 2, 1, 3, 4).reshape(batch, count, channels * window_size ** 2, search_size ** 2)
    distance = (search - search[..., search_size ** 2 // 2:search_size ** 2 // 2 + 1]).square().sum(2)
    weights = torch.exp(-distance / (channels * window_size ** 2 * 0.004))
    return weights / (weights.sum(dim=-1, keepdim=True) + 1e-5)


class SelfSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        size, stride = 128, 3
        pattern = torch.eye(stride).repeat(math.ceil(size / stride), math.ceil(size / stride))
        self.register_buffer("mask_stride", pattern[:size, :size].unsqueeze(0).unsqueeze(0))

    def forward(self, target, prediction, mask):
        predicted, expected = [], []
        for gt, image, edge in zip(target, prediction, mask):
            selected = edge.unsqueeze(0) * self.mask_stride
            if selected.sum() == 0:
                continue
            predicted.append(_similarity(image.unsqueeze(0), selected))
            expected.append(_similarity(gt.unsqueeze(0), selected))
        if not predicted:
            return prediction.sum() * 0
        predicted = torch.cat(predicted, dim=1)
        expected = torch.cat(expected, dim=1)
        kl = F.kl_div(predicted.clamp_min(1e-5).log(), expected.clamp_min(1e-5), reduction="mean")
        return 1000 * (kl + F.l1_loss(predicted, expected))
