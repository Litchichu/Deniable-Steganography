import os
import sys
import math
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from math import exp
import lpips

# The preparation for calculating LPIPS.
# Set the parameter 'verbose' as 'False' could stop printing the information of setting up LPIPS.
loss_fn = lpips.LPIPS(net='alex', verbose=False)
use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class BitWiseError:
    @staticmethod
    def cal_bwe(message, decoded_message):
        decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
                message.shape[0] * message.shape[1])
        return bitwise_avg_err


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        psnr = 0
        for i in range(img1.shape[0]):
            # torch.mean() vs. np.mean()
            mse = np.mean((img1[i, ...] - img2[i, ...]) ** 2)
            psnr += 20 * np.log10(255.0 / np.sqrt(mse))
        psnr = psnr / img1.shape[0]
        return psnr


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    # Calculate the one-dimensional Gaussian distribution vector
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    # The Gaussian kernel is created and obtained by matrix multiplication of
    # two one-dimensional Gaussian distribution vectors.
    # PS: The parameter 'channel' could be set as '3'.
    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(self, window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    # Calculate the value of SSIM
    # SSIM's formula is directly used, but when calculating the mean, the pixel average is not calculated directly,
    # the normalized Gaussian kernel convolution is used instead.
    # Calculate variance and covariance: Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
    def cal_ssim(self, img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(self, real_size, channel=channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity 对比灵敏度

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret


class LPIPS:
    def __init__(self):
        """
        Reference https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
        """
        self.name = "LPIPS"
        # if use_gpu:
        #     loss_fn.cuda()

    def cal_lpips(self, img1, img2):
        """
        Calculate the value of Learned Perceptual Image Patch Similarity, LPIPS.

        Return
        -------
        dist : torch.Tensor
        """
        if use_gpu:
            loss_fn.cuda(device)
            img1 = img1.cuda(device)
            img2 = img2.cuda(device)

        dist = sum(loss_fn.forward(img1, img2)) / img1.shape[0]
        return dist
