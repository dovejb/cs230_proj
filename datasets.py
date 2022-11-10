import os
import librosa
import soundfile as sf 
import numpy as np

import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
from constants import *

def load_train_spectrums():
    x, y = load_train_datasets()
    _, _, zx = signal.stft(x.squeeze(2), fs=SR, nperseg=512, axis=1)
    f, _, zy = signal.stft(y.squeeze(2), fs=SR, nperseg=512, axis=1)
    fstart = f.size - SPEC_SHAPE[0]
    tend = SPEC_SHAPE[1]
    return zx[:,fstart:,:tend,np.newaxis], zy[:,fstart:,:tend,np.newaxis]

# output shape: [N, freq_bin_N, time_series, 1]
def load_test_spectrums():
    x, y = load_test_datasets()
    _, _, zx = signal.stft(x.squeeze(2), fs=SR, nperseg=512, axis=1)
    f, _, zy = signal.stft(y.squeeze(2), fs=SR, nperseg=512, axis=1)
    fstart = f.size - SPEC_SHAPE[0]
    tend = SPEC_SHAPE[1]
    return zx[:,fstart:,:tend,np.newaxis], zy[:,fstart:,:tend,np.newaxis]

# output shape: [N, track, 1]
def load_train_datasets():
    x, y = load_datasets("i:/dl/train")
    return x[...,np.newaxis], y[...,np.newaxis]
def load_test_datasets():
    x, y = load_datasets("i:/dl/test")
    return x[...,np.newaxis], y[...,np.newaxis]

def load_datasets(dir, length=352800):
    files = os.listdir(dir)
    if "mix.npy" in files and "voc.npy" in files:
        with open(os.path.join(dir, "mix.npy"), "rb") as f:
            mix = np.load(f)
        with open(os.path.join(dir, "voc.npy"), "rb") as f:
            voc = np.load(f)
        return mix, voc
    mix = []
    voc = []
    for f in sorted(files):
        if "_aug_" in f: # do not use aug data now
            continue
        if not f.endswith(".wav") or f.startswith("voc"):
            continue
        mixpath = os.path.join(dir, f)
        vocpath = os.path.join(dir, "voc"+f[3:])
        m, _ = librosa.load(mixpath)
        v, _ = librosa.load(vocpath)
        if len(m) < length:
            left = (length-len(m))//2
            right = length-len(m)-left
            m = np.pad(m, (left,right))
            v = np.pad(v, (left,right))
        elif len(m) > length:
            m = m[:length]
            v = v[:length]
        mix.append(m)
        voc.append(v)
        if len(m) != length:
            print("length error", f, len(m), len(v))
            break
    
    mix, voc = np.array(mix), np.array(voc)
    with open(os.path.join(dir, "mix.npy"), "wb") as f:
        np.save(f, mix)
    with open(os.path.join(dir, "voc.npy"), "wb") as f:
        np.save(f, voc)
    return mix, voc
if __name__ == '__main__':
    if False:
        zx, zy = load_test_spectrums()
        print(zx.shape, zy.shape)
        print(zx[0,0,:])
        print(zx[9,:,0])

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