import musdb
import librosa
import numpy as np
import torch.nn.functional as F
from constants import *
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
import soundfile as sf
import metrics
import matplotlib.pyplot as plt
import time

class G:
    load_count=0
    cache={}

class Data(Dataset):
    def __init__(self, subset='train', fnum=100, n=24000, seglen=16000, rang=None) -> None:
        super().__init__()
        self.rnd = np.random.RandomState()
        self.rnd.seed(20210503)
        self.xfile = []
        self.yfile = []
        self.fnum = fnum
        self.n = n
        self.rang = rang
        self.seglen = seglen
        self.xs = []
        self.ys = []
        for i in range(fnum):
            x, _ = sf.read(f"./data/{subset}/{i+1:03}_x.wav") 
            y, _ = sf.read(f"./data/{subset}/{i+1:03}_y.wav") 
            self.xfile.append(x)
            self.yfile.append(y)

        while len(self.xs) < n:
            fno = self.rnd.randint(0, fnum)
            start = self.rnd.randint(0, len(self.xfile[fno]) - seglen)
            x = torch.Tensor(self.xfile[fno][start:start+seglen])
            y = torch.Tensor(self.yfile[fno][start:start+seglen])
            if torch.max(torch.abs(x)) < 0.00316 or torch.max(torch.abs(y)) < 0.00316: # skip silent x y (db < -50)
                continue
            self.xs.append(x)
            self.ys.append(y)

    def __len__(self):
        if self.rang is not None:
            return self.rang[1] - self.rang[0]
        return self.n
    def __getitem__(self, index) -> torch.Tensor:
        if self.rang is not None:
            idx = self.rang[0] + index
            return self.xs[idx], self.ys[idx]
        return self.xs[index], self.ys[index]

class DataModule(LightningDataModule):
    def __init__(self, n_train=24000, n_test=200, batch_size=16, seglen=16000, fnum=10, rang=None):
        super().__init__()
        self.trainset = Data('train', n=n_train, seglen=seglen, fnum=fnum, rang=rang)
        self.testset = Data('test', n=n_test, seglen=seglen, fnum=10)
        self.batch_size = batch_size
        if rang is not None and self.batch_size > rang[1]-rang[0]:
            self.batch_size = rang[1] - rang[0]
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainset, self.batch_size, True)
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.testset, self.batch_size, False)


def musdb_to_mono_and_resample(subset="train"):
    def proc(x, osr=44100, sr=SR):
        x = librosa.to_mono(x)
        x = librosa.resample(x, orig_sr=osr, target_sr=sr)
        return x
    rnd = np.random.RandomState()
    db = musdb.DB("./musdb_wav", is_wav=True, subsets=subset)
    idx = 0
    print(f"{len(db.tracks)} in total")
    for track in db.tracks:
        x = track.audio.T
        y = track.targets['vocals'].audio.T
        x, y = proc(x, track.rate), proc(y, track.rate)
        idx += 1
        print(f"Handling {idx}")
        sf.write(f"./data/{subset}/{idx:03}_x.wav", x, SR)
        sf.write(f"./data/{subset}/{idx:03}_y.wav", y, SR)

def maxdb(x:np.ndarray):
    return np.max(librosa.amplitude_to_db(x))

if __name__ == '__main__':
    musdb_to_mono_and_resample("test")
    