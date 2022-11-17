import musdb
import librosa
import numpy as np
from constants import *
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


#def mus_segments(n):
class Muset(Dataset):
    def __init__(self, n=2000, subset='train'):
        mus = musdb.DB("./musdb_wav", is_wav=True, subsets=subset)
        rnd = np.random.RandomState()
        rnd.seed(20210503)
        self.list = []
        for i in range(n):
            track = np.random.choice(mus.tracks)
            track.chunk_duration = 4.0
            track.chunk_start = np.random.uniform(0, track.duration - track.chunk_duration)
            x = librosa.to_mono(track.audio.T)
            y = librosa.to_mono(track.targets['vocals'].audio.T)
            x = librosa.resample(x, orig_sr=track.rate, target_sr=SR)
            y = librosa.resample(y, orig_sr=track.rate, target_sr=SR)
            self.list.append((x,y))
    def __len__(self):
        return len(self.list)
    def __getitem__(self, i):
        return self.list[i]

class MyDataset(LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.train = Muset()
        self.test = Muset(500, subset='test')
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, True, num_workers=1)
    def val_dataloader(self):
        return DataLoader(self.test, self.batch_size, False, num_workers=1)

if __name__ == '__main__':
    n = 5
    set = Muset(n)
    for i in range(len(set)):
        x, y = set[i]
        print("=>", i, x.shape, y.shape, np.mean(x), np.mean(y))