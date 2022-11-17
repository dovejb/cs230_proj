import numpy as np
from vae_spec import SpecVAE
import torch
from datasets import AudioSegments
from constants import *
from convtasnet.utility import sdr
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
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    scores = 10 * np.log10(num / den)
    return scores

def normalize(x):
    mean, std = torch.mean(x), torch.std(x)
    x = (x-mean)/std
    return x, mean, std

def evaluate_spec():
    ds = AudioSegments("i:/dl/test/s")
    #x, y = [], []
    for i in range(len(ds)):
        if i%10==0:
            print(f"Loading datasets, {i} of {len(ds)}")
        x, y = ds[i]
        #x.append(a)
        #y.append(b)
        if not np.any(x) or not np.any(y):
            print("Invalid", np.any(x), np.any(y))
        x = torch.from_numpy(np.array(x))
        y = torch.from_numpy(np.array(y))
        print("x", torch.any(x), x)
        print("y", torch.any(y), y)
        x, meanx, stdx = normalize(x)
        y, meany, stdy = normalize(y)
        x = x.cuda()
        y = y.cuda()
        ones = torch.ones_like(y).cuda()
        zeros = torch.zeros_like(y).cuda()
        print(x.device, y.device, ones.device, zeros.device)
        #lossfunc = ScaleInvariantSignalNoiseRatio().cuda()
        lossfunc = F.mse_loss
        print(x, y)
        print("x mean std", torch.mean(x), torch.std(x))
        print("y mean std", torch.mean(y), torch.std(y))
        print("xy", lossfunc(x, y))
        print("xx", lossfunc(x, x))
        print("yy", lossfunc(y, y))
        print("y1", lossfunc(y, ones))
        print("y0", lossfunc(y, zeros))



if __name__ == '__main__':
    evaluate_spec()