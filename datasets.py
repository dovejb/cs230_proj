import os
import librosa
import soundfile as sf 
import numpy as np

import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
from constants import *
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class AudioSegments(Dataset):
    def __init__(self, dir:str):
        files = sorted(os.listdir(dir))
        self.dir = dir
        self.files = []
        for f in files:
            if not f.startswith("x_"):
                continue
            self.files.append(f[1:])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        ff = self.files[index]
        xfn = self.dir + "/x"+ff
        yfn = self.dir + "/y"+ff
        with open(xfn, 'rb') as f:
            x = np.load(f)
        with open(yfn, 'rb') as f:
            y = np.load(f)
        return x[:WAV_SHAPE[0]], y[:WAV_SHAPE[0]]
        

class MyDataset(LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.train = AudioSegments("i:/dl/train/s")
        self.test = AudioSegments("i:/dl/test/s")
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, True, num_workers=2)
    def val_dataloader(self):
        return DataLoader(self.test, self.batch_size, False, num_workers=2)


# output shape: [N, freq_bin_N, time_series, 1]
# load music data in spectrum form
# and trim them to a good shape
def load_train_spectrums():
    x, y = load_train_datasets()
    if not TORCH:
        x = x.squeeze(2)
        y = y.squeeze(2)
    _, _, zx = signal.stft(x, fs=SR, nperseg=512, axis=1)
    f, _, zy = signal.stft(y, fs=SR, nperseg=512, axis=1)
    fstart = f.size - SPEC_SHAPE[0]
    tend = SPEC_SHAPE[1]
    if TORCH:
        return zx[:,fstart:,:tend], zy[:,fstart:,:tend]
    return zx[:,fstart:,:tend,np.newaxis], zy[:,fstart:,:tend,np.newaxis]

def load_test_spectrums():
    x, y = load_test_datasets()
    if not TORCH:
        x = x.squeeze(2)
        y = y.squeeze(2)
    _, _, zx = signal.stft(x, fs=SR, nperseg=512, axis=1)
    f, _, zy = signal.stft(y, fs=SR, nperseg=512, axis=1)
    fstart = f.size - SPEC_SHAPE[0]
    tend = SPEC_SHAPE[1]
    if TORCH:
        return zx[:,fstart:,:tend], zy[:,fstart:,:tend]
    return zx[:,fstart:,:tend,np.newaxis], zy[:,fstart:,:tend,np.newaxis]

# output shape: [N, track, 1]
# load music data in wav form
def load_train_datasets():
    x, y = load_datasets("i:/dl/train")
    if TORCH:
        # [N, C, L]
        return x[:,np.newaxis,:WAV_SHAPE[0]], y[:,np.newaxis,:WAV_SHAPE[0]]
    return x[...,np.newaxis], y[...,np.newaxis]
def load_test_datasets():
    x, y = load_datasets("i:/dl/test")
    if TORCH:
        # [N, C, L]
        return x[:,np.newaxis,:WAV_SHAPE[0]], y[:,np.newaxis,:WAV_SHAPE[0]]
        #return x, y
    return x[...,np.newaxis], y[...,np.newaxis]

# load data from disk
def load_datasets(dir, length=WAV_SHAPE[0]):
    files = os.listdir(dir)
    i=0
    for f in sorted(files):
        #if "_aug_" in f: # do not use aug data now
        #    continue
        if not f.endswith(".wav") or f.startswith("voc"):
            continue
        mixpath = os.path.join(dir, f)
        vocpath = os.path.join(dir, "voc"+f[3:])
        m, _ = librosa.load(mixpath)
        v, _ = librosa.load(vocpath)
        if not np.any(m):
            print("Invalid mix:", mixpath)
            continue
        if not np.any(v):
            print("Invalid mix:", vocpath)
            continue
        continue
        if len(m) < length:
            left = (length-len(m))//2
            right = length-len(m)-left
            m = np.pad(m, (left,right))
            v = np.pad(v, (left,right))
        elif len(m) > length:
            m = m[:length]
            v = v[:length]
        if len(m) != length:
            print("length error", f, len(m), len(v))
            break
        spath = os.path.join(dir, "s")
        if not os.path.exists(spath):
            os.makedirs(spath)
        print(f"handling {dir} - {i}")
        xfp = os.path.join(spath, f"x_{i}.npy")
        yfp = os.path.join(spath, f"y_{i}.npy")
        with open(xfp, 'wb') as f:
            np.save(f, m)
        with open(yfp, 'wb') as f:
            np.save(f, v)
        i+=1

if __name__ == '__main__':
    x, _ = librosa.load("i:/dl/test/mix_19_0_orig.wav")
    print(x)
    print(x.shape)
    print(np.any(x))
    exit()
    load_datasets("i:/dl/test")
    load_datasets("i:/dl/train")
    exit()
    #sample_rate, samples = wav.read("i:/dl/train/mix_1_2_orig.wav")
    #samples, sample_rate= sf.read("i:/dl/train/mix_1_2_orig.wav")
    samples, sample_rate= sf.read("i:/dl/A Classic Education - NightOwl.stem.wav")
    print(sample_rate)
    print(samples.shape)
    samples = np.sum(samples, axis=1)
    print(samples.shape, samples)
    f, t, Zxx = signal.stft(samples, fs=sample_rate, nperseg=1024)
    print(f.shape, t.shape, Zxx.shape)
    print(f)
    print(t)
    for i in range(10):
        print(i, np.mean(Zxx[:,i]))
    plt.pcolormesh(t, f, np.abs(Zxx), cmap='plasma')
    plt.show()
    #plt.specgram(samples.transpose(), cmap='plasma', Fs=sample_rate)