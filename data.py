import musdb
import librosa
import numpy as np
import torch.nn.functional as F
from constants import *
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
import soundfile as sf
from convtasnet.utility.sdr import calc_sdr_torch, calc_sdr
import evaluate
import matplotlib.pyplot as plt

class G:
    load_count=0
    cache={}

class Muset(Dataset):
    def __init__(self, n=20000, subset='train',postproc=None):
        """
            postproc: input (x,y) output (x,y)
        """
        self.rnd = np.random.RandomState()
        #self.rnd.seed(20210503)
        self.n = n
        self.subset = subset
        self.postproc = postproc
    def load_db(self):
        self.mus = musdb.DB("./musdb_wav", is_wav=True, subsets=self.subset)
    def generator(self, n):
        for i in range(n):
            track = self.rnd.choice(self.mus.tracks)
            track.chunk_duration = 4.0
            track.chunk_start = self.rnd.uniform(0, track.duration - track.chunk_duration)
            x = librosa.to_mono(track.audio.T)
            y = librosa.to_mono(track.targets['vocals'].audio.T)
            x = librosa.resample(x, orig_sr=track.rate, target_sr=SR)
            y = librosa.resample(y, orig_sr=track.rate, target_sr=SR)
            yield x, y
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        key = f"{self.subset}_{i}"
        if key in G.cache:
            return G.cache[key]
        xp = f"./npy/{self.subset}/x_{i:05}.npy"
        yp = f"./npy/{self.subset}/y_{i:05}.npy"
        with open(xp, 'rb') as xf, open(yp, 'rb') as yf:
            x, y = np.load(xf), np.load(yf)
        x, y = x.astype('float32'), y.astype('float32')
        if self.postproc is not None:
            x,y = self.postproc(x,y)
        G.load_count+=1
        G.cache[key] = (x,y)
        return x,y

class MusModule(LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE, n_train=20000, n_test=1000, postproc=None):
        super().__init__()
        self.batch_size = batch_size
        self.train = Muset(n_train,postproc=postproc)
        self.test = Muset(n_test, subset='test',postproc=postproc)
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, True)
    def val_dataloader(self):
        return DataLoader(self.test, self.batch_size, False)

class Single(Dataset):
    def __init__(self, idx=123, subset='train', postproc=None):
        """
            postproc: input (x,y) output (x,y)
        """
        xp = f"./npy/{subset}/x_{idx:05}.npy"
        yp = f"./npy/{subset}/y_{idx:05}.npy"
        with open(xp, 'rb') as xf, open(yp, 'rb') as yf:
            x, y = np.load(xf), np.load(yf)
        x, y = x.astype('float32'), y.astype('float32')
        self.x, self.y = x, y
        if postproc is not None:
            self.x, self.y = postproc(self.x, self.y)
        print("datasets", self.x.shape, self.y.shape)
    def __len__(self):
        return 1
    def __getitem__(self, i):
        return self.x, self.y
class SingleModule(LightningDataModule):
    def __init__(self, postproc=None, *args, **kw):
        super().__init__()
        self.dataset = Single(postproc=postproc)
    def train_dataloader(self):
        return DataLoader(self.dataset, 1)
    def val_dataloader(self):
        return DataLoader(self.dataset, 1)

def dump_wav(index=0, sub='train'):
    xp = f"./npy/{sub}/x_{index:05}.npy"
    yp = f"./npy/{sub}/y_{index:05}.npy"
    with open(xp, 'rb') as xf, open(yp, 'rb') as yf:
        x, y = np.load(xf), np.load(yf)
    print(x.shape, y.shape)
    x = x[np.newaxis,...]
    y = y[np.newaxis,...]
    print("sdr xx", calc_sdr(x,x))
    print("sdr xy", calc_sdr(x,y))
    print("sdr yx", calc_sdr(y,x))
    print("sdr yy", calc_sdr(y,y))
    x = np.squeeze(x, 0)
    y = np.squeeze(y, 0)
    sf.write("x.wav", x, SR)
    sf.write("y.wav", y, SR)


def gen_train():
    train = Muset()
    generator = train.generator(20000)
    for i,(x,y) in enumerate(generator):
        print(f"Handling train {i}")
        with open(f"./npy/train/x_{i:05}.npy", "wb") as f:
            np.save(f, x) 
        with open(f"./npy/train/y_{i:05}.npy", "wb") as f:
            np.save(f, y) 

def gen_test():
    train = Muset(subset='test')
    generator = train.generator(1000)
    for i,(x,y) in enumerate(generator):
        print(f"Handling test {i}")
        with open(f"./npy/test/x_{i:05}.npy", "wb") as f:
            np.save(f, x) 
        with open(f"./npy/test/y_{i:05}.npy", "wb") as f:
            np.save(f, y) 

def get_all():
    train = Muset(200)
    x, y = [], []
    for i in range(len(train)):
        a, b = train[i]
        x.append(a)
        y.append(b)
    x = np.array(x)
    y = np.array(y)
    return x, y

if __name__ == '__main__':
    if False:
        x, y = get_all()
        data = np.array([x, y])
        print("mean and std", np.mean(data), np.std(data))
        print(x.shape, y.shape)
    if True:
        def dataproc(x,y):
            maxlen=16000
            def proc(x):
                x = x[...,:maxlen]
                x = x.reshape(1,-1)
                x = torch.Tensor(x)
                return x
            return proc(x), proc(y)
        single = Single(postproc=dataproc)
        x, y = single[0]
        meany = torch.mean(y)
        stdy = torch.std(y)
        normy = (y-meany)/stdy
        print("meany", meany)
        print("stdy", stdy)
        print("max normy", torch.max(normy))
        print("==============initial status===========")
        print("sisnr", evaluate.sisnr(y,x))
        print("sisnri", evaluate.sisnri(y,x,x))
        print("mse", F.mse_loss(y,x))
        print("sdr", evaluate.new_sdr(y,x))
        print("==============perfect status===========")
        print("sisnr", evaluate.sisnr(y,y))
        print("sisnri", evaluate.sisnri(y,y,x))
        print("mse", F.mse_loss(y,y))
        print("sdr", evaluate.new_sdr(y,y))
        print("==============zero status===========")
        zeros = torch.zeros_like(y)
        print("sisnr", evaluate.sisnr(y,zeros))
        print("sisnri", evaluate.sisnri(y,zeros,x))
        print("mse", F.mse_loss(y,zeros))
        print("sdr", evaluate.new_sdr(y,zeros))
        yhat, _ = sf.read("./out/wave_cnn_9.wav")
        plt.figure(1)
        plt.title("yhat")
        plt.plot(yhat)
        plt.figure(2)
        plt.title("y")
        plt.plot(y.view(-1))
        plt.figure(3)
        plt.title("x")
        plt.plot(x.view(-1))
        plt.show()
