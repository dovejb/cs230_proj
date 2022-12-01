import torch.nn.functional as F
from torch import nn
import torch
import soundfile as sf
import librosa
import numpy as np

if __name__ == '__main__':
    x, _ = sf.read("./tmp/y_-9.05.wav",)
    dbs = librosa.amplitude_to_db(x)
    print(len(dbs))
    print(np.max(x), max(dbs))