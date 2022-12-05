import torch.nn.functional as F
from torch import nn
import torch
import soundfile as sf
import librosa
import numpy as np
import torchmetrics
from metrics import sisnr, new_sdr

if __name__ == '__main__':
    x, _ = sf.read("./data/train/001_x.wav",)
    x = torch.Tensor(x)
    y = x * 1.1
    tmsisdr = torchmetrics.ScaleInvariantSignalDistortionRatio()
    tmsisnr = torchmetrics.ScaleInvariantSignalNoiseRatio()
    print("sisnr", sisnr(x, y))
    print("sisnr", sisnr(y, x))
    print("tmsisdr", tmsisdr(x, y))
    print("tmsisnr", tmsisnr(x, y))

    print("sisnr", sisnr(x, x))
    print("sisnr", sisnr(x, x))
    print("tmsisdr", tmsisdr(x, x))
    print("tmsisnr", tmsisnr(x, x))

    sdr = torchmetrics.SignalDistortionRatio()
    print("sdr xx", sdr(x, x))
    print("sdr xy", sdr(x, y))
    print("sdr yx", sdr(y, x))

    print("newsdr xx", new_sdr(x, x))
    print("newsdr xy", new_sdr(x, y))
    print("newsdr yx", new_sdr(y, x))
