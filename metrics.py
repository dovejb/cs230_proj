import numpy as np
import torch
from constants import *
import torch.nn.functional as F
from torchmetrics import ScaleInvariantSignalNoiseRatio

def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    #assert references.ndim == 4
    #assert estimates.ndim == 4
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references))#, axis=(1, 2))
    den = torch.sum(torch.square(references - estimates))#, axis=(1, 2))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores

def sisnr(references:torch.Tensor, estimates:torch.Tensor, eps=1e-8):
    """
    calculate training loss
    input:
          references: reference signal, N x S tensor
          estimates: separated signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if estimates.shape != references.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                estimates.shape, references.shape))
    x_zm = estimates - torch.mean(estimates, dim=-1, keepdim=True)
    s_zm = references - torch.mean(references, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def sisnri(references:torch.Tensor, estimates:torch.Tensor, mixtures:torch.Tensor):
    return sisnr(references, estimates) - sisnr(references, mixtures)


if __name__ == '__main__':
    pass