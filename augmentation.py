import librosa
import numpy as np
import soundfile as sf
from audiomentations import Compose, PitchShift, AddGaussianNoise, TimeStretch, Shift, BandPassFilter
import os

SR = 22050

if __name__ == '__main__':
    augment = Compose([
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        #AddGaussianNoise(),
        TimeStretch(),
        Shift(min_fraction=-0.2, max_fraction=0.2),
        #BandPassFilter(p=0.3),
    ])
    dir = "i:/dl/train"
    files = os.listdir(dir)
    for f in (files):
        if f.startswith("voc") or "_aug_" in f:
            continue
        v = "voc" + f[3:]
        mixpath = os.path.join(dir, f)
        vocpath = os.path.join(dir, v)
        vocout = f"{v[:-8]}aug_9.wav"

        if vocout in files:
            print("Skip augmented file:", f)
            continue

        print("Augmenting", f)
        mix, _ = librosa.load(mixpath, sr=SR)
        voc, _ = librosa.load(vocpath, sr=SR)
        src = np.vstack((mix,voc))
        for i in range(5):
            dst = augment(src, SR)
            mixout = f"{mixpath[:-8]}aug_{i}.wav"
            vocout = f"{vocpath[:-8]}aug_{i}.wav"
            sf.write(mixout, dst[0], SR)
            sf.write(vocout, dst[1], SR)